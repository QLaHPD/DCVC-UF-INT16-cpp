// Small fixtures and tensors only. Never load/construct UF models/checkpoints.
#include "uf/archive.hpp"
#include "uf/bitstream.hpp"
#include "uf/cli.hpp"
#include "uf/entropy.hpp"
#include "uf/frame_buffer.hpp"
#include "uf/frame_source.hpp"
#include "uf/image_pixels.hpp"
#include "uf/image_archive.hpp"
#include "uf/shared_work.hpp"
#include "uf/remote.hpp"
#include "uf/stream_pipeline.hpp"
#include "uf/python_path.hpp"
#include "uf/int16_ops.hpp"
#include "uf/json_compat.hpp"
#include "uf/python_json.hpp"
#include "uf/python_numeric.hpp"
#include "pareto_random.hpp"
#ifdef UF_TEST_EVALUATION
#include "uf/evaluation_metrics.hpp"
#endif
#ifdef UF_TEST_CUDA
#include "uf/cuda_ops.hpp"
#endif
#ifdef UF_TEST_FP16
#include "uf/fp16_model.hpp"
#include "def_cutlass.h"
#include "def_elementwise.h"
#include "layers_proxy.h"
#endif
#ifdef UF_TEST_TRAINING
#include "uf/training_layers.hpp"
#include "uf/training_adamw.hpp"
#include "uf/training_recompute.hpp"
#include "uf/training_datasets.hpp"
#include "uf/training_loader.hpp"
#include <ATen/autocast_mode.h>
#include <torch/csrc/autograd/autograd.h>
#endif
#include <torch/cuda.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unistd.h>

using Json = nlohmann::json;
namespace {
void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
template<class F> void throws(F fn) {
    bool caught = false;
    try { fn(); } catch (const std::exception&) { caught = true; }
    require(caught, "Expected exception");
}
std::vector<std::uint8_t> bytes(const std::string& text) {
    return {text.begin(), text.end()};
}
void same(const torch::Tensor& actual, const torch::Tensor& expected, const std::string& label) {
    const auto cpu = actual.to(torch::kCPU);
    require(cpu.scalar_type() == expected.scalar_type() && cpu.sizes() == expected.sizes() &&
            torch::equal(cpu, expected), label + " tensor mismatch");
}
std::int64_t rounded(std::int64_t n, std::int64_t d) {
    return n < 0 ? -((-n + d / 2) / d) : (n + d / 2) / d;
}
std::int16_t saturated(std::int64_t n) {
    return static_cast<std::int16_t>(std::clamp(n, std::int64_t(-32768), std::int64_t(32767)));
}
torch::Tensor convolution_oracle(const torch::Tensor& input, const torch::Tensor& weights,
                                 const torch::Tensor& bias, int stride, int padding, int groups) {
    const auto x = input.contiguous(), w = weights.contiguous();
    const auto n = x.size(0), ci = x.size(1), hi = x.size(2), wi = x.size(3);
    const auto co = w.size(0), kh = w.size(2), kw = w.size(3);
    const auto ho = (hi + 2 * padding - kh) / stride + 1;
    const auto wo = (wi + 2 * padding - kw) / stride + 1;
    auto result = torch::empty({n, co, ho, wo}, x.options());
    const auto* xp = x.data_ptr<std::int16_t>(); const auto* wp = w.data_ptr<std::int16_t>();
    auto* out = result.data_ptr<std::int16_t>(); const auto per_group = ci / groups;
    for (int b = 0; b < n; ++b) for (int c = 0; c < co; ++c)
    for (int y = 0; y < ho; ++y) for (int z = 0; z < wo; ++z) {
        std::int64_t sum = 0; const auto first = (c / (co / groups)) * per_group;
        for (int k = 0; k < per_group; ++k) for (int yy = 0; yy < kh; ++yy)
        for (int zz = 0; zz < kw; ++zz) {
            const auto iy = y * stride + yy - padding, ix = z * stride + zz - padding;
            if (iy < 0 || ix < 0 || iy >= hi || ix >= wi) continue;
            sum += std::int64_t(xp[((b * ci + first + k) * hi + iy) * wi + ix]) *
                   wp[((c * per_group + k) * kh + yy) * kw + zz];
        }
        auto value = rounded(sum, uf::int16::weight_scale);
        if (bias.defined()) value += bias[c].item<std::int64_t>();
        out[((b * co + c) * ho + y) * wo + z] = saturated(value);
    }
    return result;
}
struct Suite {
    int passed{}, failed{};
    void run(const std::string& name, const std::function<void()>& fn) {
        try { fn(); ++passed; std::cout << "PASS " << name << std::endl; }
        catch (const std::exception& e) { ++failed; std::cout << "FAIL " << name << ": " << e.what() << std::endl; }
    }
};
void close_float(const torch::Tensor& actual,const Json& expected,const std::string& label) {
    require(torch::allclose(actual.detach().cpu().to(torch::kFloat32).flatten(),
        torch::tensor(expected.get<std::vector<float>>()),1e-5,1e-6),label);
}
#ifdef UF_TEST_EVALUATION
std::pair<torch::Tensor,torch::Tensor> metric_pair(int height,int width,int channels=0) {
    auto index=torch::arange(height*width*(channels?channels:1),torch::kFloat64);
    index=channels?index.reshape({channels,height,width}):index.reshape({height,width});
    auto original=(index*37).remainder(256);
    return {original,(original+index.remainder(7)-3).clamp(0,255)};
}
#endif
#ifdef UF_TEST_TRAINING
void same_sample(const uf::training::DatasetSample& sample,const Json& expected) {
    require(sample.size()==expected.size(),"Dataset field count");
    for(std::size_t i=0;i<sample.size();++i) {
        const auto& row=expected.at(i);auto shape=row.at("shape").get<std::vector<std::int64_t>>();torch::Tensor value;
        if(row.at("dtype")=="torch.int32")value=torch::tensor(row.at("values").get<std::vector<std::int32_t>>(),torch::kInt32);
        else value=torch::tensor(row.at("values").get<std::vector<float>>(),torch::kFloat32);
        same(sample[i],value.reshape(shape),"Dataset Python field="+std::to_string(i));
    }
}
std::vector<int> child_processes() {
    // Jetson's kernel does not expose task/children. Inspect PPIDs instead;
    // disappearing processes can be skipped while enumerating /proc.
    std::vector<int> children;
    for(const auto& entry:std::filesystem::directory_iterator("/proc")) {
        const auto name=entry.path().filename().string();
        if(name.empty() || !std::all_of(name.begin(),name.end(),[](char c){return c>='0' && c<='9';}))continue;
        std::ifstream file(entry.path()/"stat");std::string line;std::getline(file,line);
        const auto end=line.rfind(')');if(end==std::string::npos)continue;
        std::istringstream fields(line.substr(end+1));char state;int parent;
        if(fields>>state>>parent && parent==getpid())children.push_back(std::stoi(name));
    }
    std::sort(children.begin(),children.end());return children;
}
struct TestAutocastScope {
    at::DeviceType device;
    bool previous_enabled,previous_cache;
    at::ScalarType previous_dtype;
    TestAutocastScope(at::DeviceType type,at::ScalarType dtype,bool enabled,bool cache):device(type),
        previous_enabled(at::autocast::is_autocast_enabled(type)),
        previous_cache(at::autocast::is_autocast_cache_enabled()),previous_dtype(at::autocast::get_autocast_dtype(type)) {
        at::autocast::set_autocast_dtype(device,dtype);at::autocast::set_autocast_enabled(device,enabled);
        at::autocast::set_autocast_cache_enabled(cache);at::autocast::increment_nesting();
    }
    ~TestAutocastScope() {
        if(at::autocast::decrement_nesting()==0)at::autocast::clear_cache();
        at::autocast::set_autocast_dtype(device,previous_dtype);at::autocast::set_autocast_enabled(device,previous_enabled);
        at::autocast::set_autocast_cache_enabled(previous_cache);
    }
};
#endif
}

int main(int argc, char** argv) {
    if(argc==5 && std::string(argv[1])=="_prepared_transfer") {
        try {
            uf::RemoteJob job;job.url=argv[2];job.ytdlp_binary=argv[3];job.video_format="best";
            uf::PreparedDownload download(job);auto state=download.state();std::ofstream output(argv[4],std::ios::binary);
            require(bool(output),"Cannot open owned transfer output");std::array<char,8192> buffer{};
            for(;;){const auto count=read(download.input_descriptor(),buffer.data(),buffer.size());
                if(count<0)throw std::runtime_error("Cannot read prepared transfer");if(!count)break;output.write(buffer.data(),count);}
            output.close();require(!output.fail(),"Cannot close transfer output");
            uf::drain_prepared_download(download.input_descriptor(),download.error_descriptor(),*state);
            download.close();std::cout << uf::sha256_file(argv[4]) << '\n';return 0;
        }catch(const std::exception& error){std::cerr << error.what() << '\n';return 1;}
    }
#ifdef UF_TEST_TRAINING
    if(argc==2 && std::string(argv[1])=="_train_loader_worker") {
        try{return uf::training::run_training_loader_worker();}
        catch(const std::exception& error){std::cerr << error.what() << '\n';return 1;}
    }
#endif
    if (argc != 2) { std::cerr << "Fixture JSON path required\n"; return 2; }
    torch::set_num_threads(1); torch::set_num_interop_threads(1);
    std::ifstream file(argv[1]); Json fixtures; file >> fixtures;
    Suite suite;
    const auto temp = std::filesystem::temp_directory_path() / ("uf-model-free-" + std::to_string(getpid()));
    std::filesystem::create_directory(temp);
    suite.run("adaptive integer Python wire bytes", [&] {
        for (const auto& row : fixtures.at("adaptive")) {
            std::stringstream stream;
            require(uf::write_uint_adaptive(stream, row.at("value").get<std::uint32_t>()) == row.at("length"), "Write length");
            require(bytes(stream.str()) == row.at("bytes").get<std::vector<std::uint8_t>>(), "Wire bytes");
            require(uf::read_uint_adaptive(stream) == row.at("value"), "Decode value");
        }
        std::stringstream invalid; throws([&] { uf::write_uint_adaptive(invalid, 1u << 30); });
        std::stringstream truncated(std::string(1, '\xc0')); throws([&] { uf::read_uint_adaptive(truncated); });
    });
    suite.run("SPS and packet Python wire bytes", [&] {
        std::stringstream stream; uf::write_sps(stream, {3,16384,128});
        std::vector<std::uint8_t> payload(17); for (int i=0;i<17;++i) payload[i]=i;
        uf::write_packet(stream, {uf::NalType::intra,3,63,1,false,payload});
        uf::write_packet(stream, {uf::NalType::predicted,3,0,1,true,{255,0,128}});
        require(bytes(stream.str()) == fixtures.at("packet_bytes").get<std::vector<std::uint8_t>>(), "SPS/packet bytes");
        auto header=uf::read_header(stream); require(header.type==uf::NalType::sps && header.sps_id==3,"SPS header");
        auto sps=uf::read_sps(stream,3); require(sps.width==16384 && sps.height==128,"SPS dimensions");
        header=uf::read_header(stream); auto p=uf::read_packet(stream,header.type,header.sps_id);
        require(p.payload==payload && p.qp==63,"Intra packet");
        header=uf::read_header(stream); p=uf::read_packet(stream,header.type,header.sps_id);
        require(p.reset_feature_memory && p.payload==std::vector<std::uint8_t>({255,0,128}),"Predicted packet");
        require(!uf::read_optional_header(stream),"Clean EOF");
    });
    suite.run("UF16 preamble and SPS registry lifetime", [&] {
        uf::Preamble p; p.integer=true; p.image_identity.fill(17); p.video_identity.fill(29);
        std::stringstream stream; uf::write_preamble(stream,p); auto decoded=uf::read_preamble(stream);
        require(decoded.integer && decoded.image_identity==p.image_identity && decoded.video_identity==p.video_identity,"Preamble");
        uf::SpsHelper helper; auto allocated=helper.get_sps_id({9,64,48}); require(allocated.first==0 && allocated.second,"Allocate SPS");
        auto retained=helper.get_sps_by_id(0); helper.add_sps_by_id({0,128,96});
        require(retained->width==64 && helper.get_sps_by_id(0)->width==128,"Retained lookup lifetime");
    });
    suite.run("Python Random seed, draws, shuffle and Gaussian", [&] {
        for (const auto& row:fixtures.at("random")) {
            auto seed=uf::PythonInteger(row.at("seed").get<std::string>());
            uf::pareto_detail::PythonRandom rng(seed);
            for (const auto& expected:row.at("draws")) require(rng.random()==expected.get<double>(),"Random draw");
            rng=uf::pareto_detail::PythonRandom(seed);
            for (std::size_t i=0;i<row.at("bounds").size();++i) require(rng.below(row["bounds"][i].get<std::uint64_t>())==row["choices"][i],"Random below");
            rng=uf::pareto_detail::PythonRandom(seed); std::vector<int> values(19); for(int i=0;i<19;++i)values[i]=i; rng.shuffle(values);
            require(values==row.at("shuffled").get<std::vector<int>>(),"Shuffle");
            rng=uf::pareto_detail::PythonRandom(seed);
            for (const auto& expected:row.at("gaussian")) require(std::abs(rng.gauss(1.25,0.75)-expected.get<double>())<1e-13,"Gaussian");
        }
    });
    suite.run("UF rANS Python extension wire parity", [&] {
        for (const auto& row:fixtures.at("entropy")) {
            uf::EntropyEncoder encoder; uf::EntropyDecoder decoder;
            auto cdf=row.at("cdf").get<std::vector<std::int32_t>>(); auto lengths=row.at("lengths").get<std::vector<std::int32_t>>();
            for(int i=0;i<2;++i){encoder.set_cdf(i,cdf,lengths);decoder.set_cdf(i,cdf,lengths);}
            auto z=row.at("z").get<std::vector<std::int8_t>>(); auto y=row.at("y").get<std::vector<std::int8_t>>();
            encoder.reset();encoder.encode_y(row.at("packed").get<std::vector<std::int16_t>>());encoder.encode_z(z,0,2);
            auto output=encoder.flush();require(output==row.at("bytes").get<std::vector<std::uint8_t>>(),"Encoded rANS bytes");
            decoder.set_stream(output);require(decoder.decode_z(z.size(),0,2)==z,"Decode Z");
            require(decoder.decode_y(row.at("indexes").get<std::vector<std::uint8_t>>())==y,"Decode Y");
        }
    });
    suite.run("JSON exact numbers, Unicode and numeric syntax", [&] {
        auto value=uf::parse_python_json("{\"b\": true, \"a\": 18446744073709551616, \"s\": \"\\ud83d\\ude00\"}");
        require(uf::python_canonical_json(value)=="{\"a\":18446744073709551616,\"b\":true,\"s\":\"😀\"}","Canonical JSON");
        require(uf::json_python_equal(Json(1),Json(true)),"Bool/integer equality");
        require(!uf::json_python_equal(uf::parse_python_json("9007199254740993"),Json(9007199254740992.0)),"Exact integer/float equality");
        require(uf::PythonInteger("  +１２_３  ").decimal()=="123","Unicode integer");
        require(uf::python_float(" １２.５ ")==12.5,"Unicode float");
        require((uf::PythonInteger("999999999999999999999")*uf::PythonInteger(2)).decimal()=="1999999999999999999998","Big multiplication");
    });
    suite.run("owned file copy and PNG round trip", [&] {
        auto src=temp/"source", dst=temp/"copy"; {std::ofstream f(src);f << "test payload\n";}
        uf::copy_completed_archive_file(src,dst);require(uf::sha256_file(src)==uf::sha256_file(dst),"Copy hash");
        throws([&]{uf::copy_completed_archive_file(src,dst);});
        uf::ImagePixels image{2,2,{255,0,0,0,255,0,0,0,255,19,20,21}};auto path=temp/"rgb.png";
        uf::write_image_png(path,image);uf::verify_image(path);auto actual=uf::read_image_rgb(path);
        require(actual.width==2 && actual.height==2 && actual.rgb==image.rgb,"PNG pixels");
    });
    for(const auto& row:fixtures.at("raster_images"))suite.run("raster decoder Pillow RGB/orientation parity "+row.at("name").get<std::string>(),[&,row]{
        const auto path=std::filesystem::path(row.at("path").get<std::string>());uf::verify_image(path);
        for(const auto& expected:row.at("expected")) {
            auto actual=uf::read_image_rgb(path,expected.at("oriented").get<bool>());
            require(actual.width==expected.at("width").get<std::uint32_t>() && actual.height==expected.at("height").get<std::uint32_t>(),"Raster dimensions/orientation");
            require(actual.rgb==expected.at("rgb").get<std::vector<std::uint8_t>>(),"Pillow decoded RGB bytes");
        }
    });
    suite.run("raster malformed/truncated inputs and embedded NUL rejection",[&]{
        auto path=temp/"truncated.png";{std::ofstream file(path,std::ios::binary);file.write("\211PNG\r\n\032\n",8);}
        throws([&]{uf::verify_image(path);});throws([&]{uf::read_image_rgb(path);});
        path=temp/"bad.bmp";{std::ofstream file(path,std::ios::binary);file.write("BM\0\0",4);}
        throws([&]{uf::read_image_rgb(path);});
        auto nul=std::filesystem::path(std::string("bad\0image",9));throws([&]{uf::verify_image(nul);});throws([&]{uf::read_image_rgb(nul);});
    });
    suite.run("atomic JSON replacement and Unicode path round trip",[&]{
        auto path=temp/"状态.json";Json first={{"step",1},{"text","😀"}},second={{"step",2},{"text","完成"}};
        uf::write_python_json_file_atomic(path,first);require(uf::json_python_equal(uf::read_python_json_file(path,"utf-8",false),first),"First publication");
        uf::write_python_json_file_atomic(path,second);require(uf::json_python_equal(uf::read_python_json_file(path,"utf-8",false),second),"Atomic replacement");
        require(uf::python_path_from_text(uf::python_path_text(path))==path,"Unicode filename");
    });
    suite.run("viewer display RGB and crop endpoints Python parity",[&]{
        for(const auto& row:fixtures.at("viewer_display")) {
            auto actual=uf::display_rgb(row.at("raw").get<std::vector<std::uint8_t>>(),row.at("width").get<int>(),row.at("height").get<int>(),
                row.at("crop_width").get<int>(),row.at("crop_height").get<int>());
            require(actual==row.at("rgb").get<std::vector<std::uint8_t>>(),"Display RGB bytes/cropping");
        }
        for(const auto& row:fixtures.at("viewer_crop"))require(uf::image_crop_extent(uf::parse_python_json(row.at("stop").get<std::string>()),17)==row.at("expected").get<int>(),"Image slice endpoints");
        throws([&]{uf::image_crop_extent(Json(5.0),17);});throws([&]{uf::image_crop_extent(Json("5"),17);});
        throws([&]{uf::display_rgb(std::vector<std::uint8_t>(18),3,4,3,4);});
        throws([&]{uf::display_rgb(std::vector<std::uint8_t>(48),8,4,9,4);});
    });
    for(const auto& row:fixtures.at("viewer_index"))suite.run("viewer lazy packet index Python parity "+row.at("runtime").get<std::string>()+" "+row.at("variant").get<std::string>(),[&,row]{
        auto root=temp/("viewer-"+row.at("runtime").get<std::string>()+"-"+row.at("variant").get<std::string>());std::filesystem::create_directory(root);
        auto payload=root/"video.bin";{std::ofstream file(payload,std::ios::binary);auto values=row.at("bytes").get<std::vector<std::uint8_t>>();file.write(reinterpret_cast<const char*>(values.data()),values.size());}
        Json pipeline={{"runtime",row.at("runtime")},{"variant",row.at("variant")}};
        if(row.at("runtime")=="int16") {
            std::string image_id,video_id;for(int i=0;i<32;++i){image_id+="07";video_id+="09";}
            pipeline["integer"]={{"image",image_id},{"video",video_id}};
        }
        Json metadata={{"format","dcvc-uf-archive"},{"version",1},{"pipeline",pipeline},{"video",{{"width",8},{"height",4},{"frames",row.at("frames")},{"packets",4}}},
            {"validation",{{"mode","artifact-hashes"},{"decode_performed",false}}},{"artifacts",{{"video.bin",{{"size",std::filesystem::file_size(payload)},{"sha256",uf::sha256_file(payload)}}}}}};
        uf::write_python_json_file_atomic(root/"manifest.json",metadata);int calls=0;
        uf::IndexedFrameSource source(root,[&](const Json&,bool){++calls;return std::shared_ptr<uf::Codec>{};});
        require(calls==0 && source.frame_count()==row.at("frames").get<int>(),"Index eagerly loaded a codec");
        const auto& packets=source.packets();require(packets.size()==row.at("packets").size(),"Index packet count");
        for(std::size_t i=0;i<packets.size();++i){const auto& actual=packets[i];const auto& expected=row.at("packets").at(i);
            require(actual.offset==expected.at("offset").get<std::int64_t>() && actual.first==expected.at("first").get<std::uint64_t>() && actual.count==expected.at("count").get<std::uint64_t>() &&
                static_cast<int>(actual.type)==expected.at("nal_type").get<int>() && actual.qp==expected.at("qp").get<int>() && actual.sps.id==expected.at("sps").at("sps_id").get<int>(),"Indexed packet metadata");}
        throws([&]{source.seek(source.frame_count());});throws([&]{source.seek(0,[]{return true;});});require(calls==0,"Invalid/canceled seek invoked loader");
        bool missing=false;try{source.seek(0);}catch(const std::runtime_error& error){missing=std::string(error.what())=="UF viewer model loader returned no codec";}
        require(missing && calls==1,"Lazy loader boundary");source.close_checked();source.close_checked();
        metadata["video"]["packets"]=5;uf::write_python_json_file_atomic(root/"manifest.json",metadata);
        throws([&]{uf::IndexedFrameSource invalid(root,{});});
    });
    for(const auto& row:fixtures.at("viewer_images"))suite.run("viewer image header index and ordered pipeline repr "+row.at("runtime").get<std::string>(),[&,row]{
        auto path=temp/("viewer-image-"+row.at("runtime").get<std::string>()+".dcvci");
        auto header=row.at("metadata_json").get<std::string>();auto payload=row.at("bytes").get<std::vector<std::uint8_t>>();
        {std::ofstream file(path,std::ios::binary);file.write("UFIMAGE\1",8);const auto size=static_cast<std::uint32_t>(header.size());
         for(int shift:{24,16,8,0})file.put(static_cast<char>(size>>shift));file.write(header.data(),header.size());file.write(reinterpret_cast<const char*>(payload.data()),payload.size());}
        int calls=0;bool image_only=false;uf::IndexedFrameSource source(path,[&](const Json&,bool image){++calls;image_only=image;return std::shared_ptr<uf::Codec>{};});
        require(calls==0 && source.frame_count()==1 && source.packets().size()==1 && source.packets()[0].type==uf::NalType::intra,"Image lazy indexing");
        require(source.pipeline_repr()==row.at("pipeline_repr").get<std::string>(),"Original ordered pipeline repr");
        throws([&]{source.seek(1);});throws([&]{source.seek(0,[]{return true;});});require(calls==0,"Invalid image seek loaded codec");
        throws([&]{source.seek(0);});require(calls==1 && image_only,"Image-only lazy loader boundary");source.close_checked();source.close_checked();
    });
    suite.run("archive artifact integrity and unsafe-path rejection",[&]{
        auto root=temp/"bundle";std::filesystem::create_directory(root);auto payload=root/"video.bin";
        {std::ofstream out(payload);out << "synthetic artifact";}
        Json metadata={{"format","dcvc-uf-archive"},{"version",1},{"pipeline",{{"runtime","int16"}}},
            {"validation",{{"mode","artifact-hashes"},{"decode_performed",false}}},
            {"artifacts",{{"video.bin",{{"size",std::filesystem::file_size(payload)},{"sha256",uf::sha256_file(payload)}}}}}};
        auto manifest=root/"manifest.json";uf::write_python_json_file_atomic(manifest,metadata);
        auto archive=uf::load_archive(root);require(uf::artifact_path(archive,"video.bin")==payload,"Artifact path");
        Json wrong_pipeline={{"runtime","fp16"}};throws([&]{uf::load_archive(root,true,&wrong_pipeline);});
        {std::ofstream out(payload,std::ios::app);out << "changed";}throws([&]{uf::load_archive(root);});
        uf::load_archive(root,false); // Metadata-only inspection intentionally skips hashes.
        metadata["files"]={{"video.bin","../outside"}};uf::write_python_json_file_atomic(manifest,metadata);throws([&]{uf::load_archive(root,false);});
        metadata.erase("files");metadata["validation"]={{"mode","artifact-hashes"},{"decode_performed",true}};
        uf::write_python_json_file_atomic(manifest,metadata);throws([&]{uf::load_archive(root,false);});
    });
    suite.run("flat manifest mapping and symlink rejection",[&]{
        auto root=temp/"flat";std::filesystem::create_directory(root);auto payload=root/"clip.bin";
        {std::ofstream out(payload);out << "synthetic";}
        Json metadata={{"format","dcvc-uf-archive"},{"version",1},{"layout","flat"},{"archive_stem","clip"},
            {"pipeline",{{"runtime","int16"}}},{"validation",{{"mode","artifact-hashes"},{"decode_performed",false}}},
            {"files",{{"video.bin","clip.bin"}}},{"artifacts",{{"video.bin",{{"size",9},{"sha256",uf::sha256_file(payload)}}}}}};
        auto manifest=root/"clip.uf.json";uf::write_python_json_file_atomic(manifest,metadata);
        auto archive=uf::load_archive(payload);require(archive.manifest==manifest,"Flat sidecar selection");
        metadata["files"]["extra"]="clip.bin";uf::write_python_json_file_atomic(manifest,metadata);throws([&]{uf::load_archive(payload,false);});
        metadata["files"].erase("extra");uf::write_python_json_file_atomic(manifest,metadata);
        std::filesystem::remove(payload);std::filesystem::create_symlink(temp/"source",payload);throws([&]{uf::load_archive(payload);});
    });
    suite.run("intra archive payload hashing and retention",[&]{
        auto payload=temp/"intra.payload";{std::ofstream out(payload,std::ios::binary);out << "tiny opaque packet";}
        Json metadata={{"format","dcvc-uf-image"},{"pipeline",{{"image_only",true},{"runtime","int16"}}},
            {"video",{{"frames",1},{"packets",1}}},{"validation",{{"mode","artifact-hashes"},{"decode_performed",false}}},
            {"payload_size",std::filesystem::file_size(payload)},{"payload_sha256",uf::sha256_file(payload)}};
        auto header=uf::python_canonical_json(metadata);auto path=temp/"tiny.dcvci";
        {std::ofstream out(path,std::ios::binary);out.write("UFIMAGE\x01",8);auto length=static_cast<std::uint32_t>(header.size());
            for(int shift:{24,16,8,0})out.put(static_cast<char>((length>>shift)&255));out << header << "tiny opaque packet";}
        auto retained=uf::read_intra_image(path,true,true),streamed=uf::read_intra_image(path,true,false);
        require(retained.payload==bytes("tiny opaque packet") && streamed.payload.empty(),"Payload retention");
        require(retained.payload_offset==12+header.size(),"Payload offset");
        {std::ofstream out(path,std::ios::binary|std::ios::app);out.put('x');}throws([&]{uf::read_intra_image(path,true,false);});
    });
    suite.run("shared lease ownership, heartbeat and completion",[&]{
        auto channel=temp/"channel";std::filesystem::create_directory(channel);
        Json config=fixtures.at("shared_pipeline_config");
        uf::SharedWorkPool first(uf::WorkKind::video,30,"first"),second(uf::WorkKind::video,30,"second");
        auto id=first.ensure_pipeline(channel,config);require(id==second.ensure_pipeline(channel,config),"Pipeline agreement");
        require(id==fixtures.at("shared_pipeline_id"),"Python/native pipeline fingerprint");
        auto lease=first.try_acquire(channel,"video");require(bool(lease),"First lease");
        std::optional<uf::SharedClaimInfo> owner;require(!second.try_acquire(channel,"video",&owner) && owner,"Busy owner");
        lease->heartbeat({{"frames",7}});auto claim=second.read_claim(channel,"video");require(claim && claim->progress.at("frames")==7,"Heartbeat progress");
        {std::ofstream out(channel/"payload.bin");out << "artifact";}
        lease->mark_done({"payload.bin"});auto complete=second.completed_artifacts(channel,"video");
        require(complete && *complete==std::vector<std::string>{"payload.bin"},"Completion evidence");
        lease->release();lease->release();require(!std::filesystem::exists(first.claim_path(channel,"video")),"Claim released");
        auto next=second.try_acquire(channel,"video");require(bool(next),"Reacquire after release");next->release();
        std::filesystem::resize_file(channel/"payload.bin",0);require(!first.completed_artifacts(channel,"video"),"Empty artifact invalidates completion");
        auto changed=config;changed["qp"]=37;throws([&]{second.ensure_pipeline(channel,changed);});
    });
    suite.run("shared stale claim recovery",[&]{
        auto channel=temp/"stale-channel";std::filesystem::create_directory(channel);
        uf::SharedWorkPool pool(uf::WorkKind::video,30,"recover");pool.ensure_pipeline(channel,{{"runtime","int16"}});
        auto claim=pool.claim_path(channel,"abandoned");std::filesystem::create_directory(claim);
        auto state=claim/"state.json";uf::write_python_json_file_atomic(state,{{"owner","gone"},{"pid",0}});
        std::filesystem::last_write_time(state,std::filesystem::file_time_type::clock::now()-std::chrono::seconds(60));
        auto lease=pool.try_acquire(channel,"abandoned");require(lease && lease->recovered_stale(),"Recover stale claim");lease->check();lease->release();
    });
    suite.run("shared ownership loss preserves successor claim",[&]{
        auto channel=temp/"ownership-channel";std::filesystem::create_directory(channel);
        uf::SharedWorkPool pool(uf::WorkKind::video,30,"original");pool.ensure_pipeline(channel,{{"runtime","int16"}});
        auto lease=pool.try_acquire(channel,"video");require(bool(lease),"Original lease");
        auto claim=pool.claim_path(channel,"video"),old=channel/"old-claim";
        std::filesystem::rename(claim,old);std::filesystem::create_directory(claim);
        auto state=claim/"state.json";uf::write_python_json_file_atomic(state,{{"owner","successor"},{"token","different"},{"pid",getpid()}});
        throws([&]{lease->check();});lease->release();
        require(std::filesystem::is_directory(claim) && std::filesystem::is_regular_file(state),"Original release removed successor");
    });
    suite.run("shared lease checked finally releases after body error",[&]{
        auto channel=temp/"finally-channel";std::filesystem::create_directory(channel);
        uf::SharedWorkPool pool(uf::WorkKind::video,30,"finally");pool.ensure_pipeline(channel,{{"runtime","int16"}});
        auto lease=pool.try_acquire(channel,"video");require(bool(lease),"Lease");
        bool body=false;
        try{uf::with_shared_lease(*lease,[]{throw std::runtime_error("synthetic body failure");});}
        catch(const std::runtime_error& error){body=std::string(error.what())=="synthetic body failure";}
        require(body && !std::filesystem::exists(pool.claim_path(channel,"video")),"Finally cleanup/body exception");
        auto replacement=pool.try_acquire(channel,"video");require(bool(replacement),"Lease remained busy");replacement->release();
    });
    suite.run("private cookie copy normal and exceptional cleanup",[&]{
        auto source=temp/"cookies.txt";{std::ofstream file(source);file << "synthetic cookie\n";}
        std::filesystem::path copied;
        {uf::RemoteCookies cookies(source);require(cookies.path() && *cookies.path()!=source,"Private cookie path");
         copied=*cookies.path();require(uf::sha256_file(source)==uf::sha256_file(copied),"Cookie contents");cookies.close();cookies.close();}
        require(!std::filesystem::exists(copied) && std::filesystem::exists(source),"Cookie cleanup preserved original");
        {uf::RemoteCookies cookies(source);copied=*cookies.path();throws([&]{uf::with_remote_cookies(cookies,[]{throw std::runtime_error("synthetic cookie body");});});}
        require(!std::filesystem::exists(copied),"Exceptional cookie cleanup");
    });
    suite.run("local yt-dlp metadata argv, selection, errors and cancellation",[&]{
        auto executable=temp/"fake-ytdlp.py",config=temp/"fake-ytdlp.json",calls=temp/"fake-ytdlp.calls";
        {std::ofstream script(executable);script << R"PY(#!/usr/bin/python3
import json,sys,time
from pathlib import Path
root=Path(__file__).parent
configuration=json.loads((root/'fake-ytdlp.json').read_text())
arguments=sys.argv[1:]
cookie=arguments[arguments.index('--cookies')+1] if '--cookies' in arguments else None
with (root/'fake-ytdlp.calls').open('a') as log:
    log.write(json.dumps({'args':arguments,'cookie':cookie,'contents':Path(cookie).read_text() if cookie else None})+'\n')
flat='--flat-playlist' in arguments
if '--dump-single-json' not in arguments:
    if configuration.get('block_download'):
        (root/'fake-ytdlp.started').write_text('started')
        time.sleep(60)
    if configuration.get('download_error'):
        sys.stderr.write('synthetic download failure');sys.exit(9)
    sys.stdout.buffer.write(bytes(range(64))*3)
    sys.stderr.write('UF_DOWNLOAD:1:1:192:192:192:finished\n')
    sys.exit(0)
if not flat and configuration.get('error'):
    sys.stderr.write('synthetic metadata failure 😀');sys.exit(7)
print(json.dumps(configuration['listing' if flat else 'info']))
)PY";}
        std::filesystem::permissions(executable,std::filesystem::perms::owner_all);
        const std::string video="ABCDEFGHIJK",channel="UC"+std::string(22,'A'),url="https://www.youtube.com/watch?v="+video;
        Json entry={{"id",video},{"url",url},{"channel_id",channel}};
        Json info={{"id",video},{"webpage_url",url},{"channel_id",channel},{"duration",12.0},{"upload_date","20261006"},{"is_live",false},
            {"formats",Json::array({
              {{"format_id","v720"},{"url","https://synthetic.invalid/v720"},{"vcodec","h264"},{"acodec","none"},{"width",1280},{"height",720},{"fps",30},{"tbr",1000}},
              {{"format_id","v480"},{"url","https://synthetic.invalid/v480"},{"vcodec","h264"},{"acodec","none"},{"width",640},{"height",480},{"fps",25},{"tbr",500}},
              {{"format_id","audio"},{"url","https://synthetic.invalid/audio"},{"vcodec","none"},{"acodec","opus"},{"abr",48},{"asr",48000}}
            })}};
        Json settings={{"listing",{{"_type","playlist"},{"entries",Json::array({entry,entry})}}},{"info",info}};
        uf::write_python_json_file_atomic(config,settings);
        auto source=temp/"remote-cookies.txt";{std::ofstream file(source);file << "synthetic only";}
        uf::RemoteOptions options;options.source_urls={url};options.output_root=temp/"remote-output";
        options.ytdlp_binary=executable.string();options.cookies=source;options.write_thumbnail=false;options.max_height=480;
        auto result=uf::discover_remote_jobs(options);require(result.jobs.size()==1 && result.failed==0,"Remote dedup/discovery");
        const auto& job=result.jobs[0];require(job.video_id==video && job.channel_id==channel && job.video_format=="v480" && job.audio_format=="audio","Format/provider selection");
        require(job.media.width==640 && job.media.height==480 && job.media.fps==25 && job.media.duration==12,"Remote media metadata");
        {std::ifstream log(calls);std::string line;int count=0;
         while(std::getline(log,line)){auto row=Json::parse(line);auto arguments=row["args"].get<std::vector<std::string>>();
            require(std::find(arguments.begin(),arguments.end(),"--ignore-config")!=arguments.end() &&
                    std::find(arguments.begin(),arguments.end(),"--skip-download")!=arguments.end(),"Metadata argv");
            require(arguments.size()>=2 && arguments[arguments.size()-2]=="--" && arguments.back()==url,"URL argv separator");
            require(row["contents"]=="synthetic only" && row["cookie"]!=source.string() &&
                    !std::filesystem::exists(row["cookie"].get<std::string>()),"Copied-cookie lifecycle");++count;}
         require(count==2,"Duplicate entry launched extra metadata call");}
        settings["info"]["is_live"]=true;uf::write_python_json_file_atomic(config,settings);
        auto live=uf::discover_remote_jobs(options);require(live.jobs.empty() && live.skipped_live==1,"Live video skipped");
        settings["info"]["is_live"]=false;settings["error"]=true;uf::write_python_json_file_atomic(config,settings);
        auto failed=uf::discover_remote_jobs(options);require(failed.jobs.empty() && failed.failed==1,"Metadata error accounted");
        auto previous_size=std::filesystem::file_size(calls);auto canceled=uf::discover_remote_jobs(options,{},[]{return true;});
        require(canceled.jobs.empty() && std::filesystem::file_size(calls)==previous_size,"Canceled metadata launched child");
    });
    suite.run("prepared fake download completion and checked close",[&]{
        uf::RemoteJob job;job.video_id="ABCDEFGHIJK";job.channel_id="UC"+std::string(22,'A');
        job.url="https://synthetic.invalid/video";job.video_format="v480";
        job.ytdlp_binary=(temp/"fake-ytdlp.py").string();job.cookies=temp/"remote-cookies.txt";
        uf::PreparedDownload download(job);auto state=download.state();
        uf::drain_prepared_download(download.input_descriptor(),download.error_descriptor(),*state);
        require(state->status()==0,"Prepared download status");download.close();download.close();
        require(state->status()==0,"Shared state lifetime after close");
    });
    suite.run("prepared fake download error propagation",[&]{
        auto config=temp/"fake-ytdlp.json";auto settings=uf::read_python_json_file(config,"utf-8",false);
        settings["download_error"]=true;uf::write_python_json_file_atomic(config,settings);
        uf::RemoteJob job;job.video_id="ABCDEFGHIJK";job.channel_id="UC"+std::string(22,'A');
        job.url="https://synthetic.invalid/video";job.video_format="v480";
        job.ytdlp_binary=(temp/"fake-ytdlp.py").string();job.cookies=temp/"remote-cookies.txt";
        uf::PreparedDownload download(job);auto state=download.state();
        bool caught=false;
        try { uf::drain_prepared_download(download.input_descriptor(),download.error_descriptor(),*state); }
        catch(const std::runtime_error& error) {
            caught=std::string(error.what()).find("synthetic download failure")!=std::string::npos;
        }
        require(caught,"Download error diagnostic was not propagated");
        require(state->status()==9,"Prepared nonzero status");download.close();
    });
    suite.run("prepared active download cancellation and cookie cleanup",[&]{
        auto config=temp/"fake-ytdlp.json";auto settings=uf::read_python_json_file(config,"utf-8",false);
        settings["download_error"]=false;settings["block_download"]=true;uf::write_python_json_file_atomic(config,settings);
        uf::RemoteJob job;job.video_id="ABCDEFGHIJK";job.channel_id="UC"+std::string(22,'A');
        job.url="https://synthetic.invalid/video";job.video_format="v480";
        job.ytdlp_binary=(temp/"fake-ytdlp.py").string();job.cookies=temp/"remote-cookies.txt";
        uf::PreparedDownload download(job);auto state=download.state();
        auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
        while(!std::filesystem::exists(temp/"fake-ytdlp.started") && std::chrono::steady_clock::now()<deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        require(std::filesystem::exists(temp/"fake-ytdlp.started"),"Fake child did not start");
        require(state->status()==uf::SharedStreamState::pending,"Blocking child completed early");
        std::filesystem::path copied;{std::ifstream log(temp/"fake-ytdlp.calls");std::string line;
            while(std::getline(log,line))copied=Json::parse(line)["cookie"].get<std::string>();}
        require(std::filesystem::exists(copied),"Private cookie missing during download");
        download.close();download.close();
        require(state->status()!=uf::SharedStreamState::pending && state->status()!=0,"Canceled child status");
        require(download.input_descriptor()==-1 && download.error_descriptor()==-1,"Closed download descriptors");
        require(!std::filesystem::exists(copied) && std::filesystem::exists(*job.cookies),"Cancel cookie cleanup");
    });
    suite.run("shared stream descriptor aliases and status",[&]{
        uf::SharedStreamState owner;uf::SharedStreamState alias(owner.descriptor());
        require(owner.status()==uf::SharedStreamState::pending && !alias.near_end(),"Initial shared state");
        owner.set_status(-7);require(alias.status()==-7,"Shared status alias");alias.set_near_end();require(owner.near_end(),"Shared near-end flag");
        owner.claim();owner.release();require(alias.capacity()>=1,"Shared stream capacity");
    });
    suite.run("shared stream calibration thresholds and one-time update",[&]{
        uf::SharedStreamState owner;uf::SharedStreamState alias(owner.descriptor());
        require(!owner.observe(150,30),"Unclaimed stream calibrated");owner.claim();
        require(!owner.observe(0,30) && !owner.observe(150,0) && !owner.observe(149,30),"Invalid/short calibration accepted");
        auto capacity=owner.observe(150,30);require(capacity && *capacity==60 && alias.capacity()==60,"Shared calibrated capacity");
        require(!alias.observe(150,25) && owner.capacity()==60,"Calibration repeated");owner.release();
    });
    suite.run("bounded prefetch order, EOF and close", [&] {
        auto index=std::make_shared<int>(0); bool canceled=false;
        uf::BufferedFrames buffer([index]() -> std::optional<std::vector<std::uint8_t>> {
            if(*index==10)return std::nullopt;return std::vector<std::uint8_t>{static_cast<std::uint8_t>((*index)++)};
        },[&]{canceled=true;},3,1);
        buffer.prime();buffer.set_capacity(2);for(int i=0;i<10;++i){auto item=buffer.read();require(item && item->at(0)==i,"Prefetch order");}
        require(!buffer.read(),"Prefetch EOF");buffer.close_checked();buffer.close_checked();require(canceled,"Cancel on close");throws([&]{buffer.read();});
    });
    suite.run("subprocess dual-pipe capture", [&] {
        uf::ChildPipe child({"/usr/bin/python3","-c",
            "import sys;sys.stdout.write('x'*131072);sys.stderr.write('y'*131072)"},
            std::nullopt,false,std::nullopt,false,false,true);
        auto output=child.communicate();child.close_checked();
        require(output.returncode==0 && output.output==std::string(131072,'x') &&
                output.errors==std::string(131072,'y'),"Concurrent stdout/stderr capture");
    });
    suite.run("FFmpeg probe, buffered frames and early close", [&] {
        auto path=temp/"tiny.mkv";
        uf::ChildPipe generate({"ffmpeg","-v","error","-nostdin","-f","lavfi","-i",
            "testsrc=size=8x4:rate=5","-frames:v","6","-threads","1","-pix_fmt",
            "yuv420p","-c:v","ffv1","-y",path.string()},
            std::nullopt,false,std::nullopt,false,false,true);
        auto output=generate.communicate();generate.close_checked();
        require(output.returncode==0,"Generate video: "+output.errors);
        const auto info=uf::probe_media(path);require(info.width==8 && info.height==4 && info.fps==5 && !info.audio,"Probe media");
        uf::BufferedFrameReader reader(path,8,4,info,std::nullopt,1,std::nullopt,"none",3);
        reader.prime();int count=0;while(auto frame=reader.read()){require(frame->size()==96,"Expanded YUV444 bytes");++count;}
        require(count==6,"Frame count");reader.close_checked();reader.close_checked();
        uf::BufferedFrameReader partial(path,8,4,info,std::nullopt,1,std::nullopt,"none",2);
        partial.prime();require(bool(partial.read()),"First frame");partial.close_checked();
    });
    suite.run("CPU integer byte conversion and ties", [&] {
        auto values=torch::arange(256,torch::kInt64).to(torch::kUInt8);auto expected=torch::empty({256},torch::kInt16);
        for(int i=0;i<256;++i)expected.data_ptr<std::int16_t>()[i]=saturated(rounded((2*i-255)*512,510));
        same(uf::int16::from_bytes(values),expected,"CPU byte conversion");
        auto ties=torch::tensor({-769,-768,-767,-257,-256,-255,255,256,257,767,768,769},torch::kInt64);
        auto rounded_values=torch::empty(ties.sizes(),torch::kInt64);
        for(int i=0;i<ties.numel();++i)rounded_values.data_ptr<std::int64_t>()[i]=rounded(ties.data_ptr<std::int64_t>()[i],512);
        same(uf::int16::divide_round(ties,512),rounded_values,"Ties away from zero");
    });
    const bool cuda=uf::int16::cuda_compiled() && torch::cuda::is_available();
    suite.run("CUDA backend available",[&]{require(cuda,"CUDA backend unavailable on expected Orin");});
    if(cuda) {
        suite.run("CPU/CUDA integer elementwise parity",[&]{
            auto values=((torch::arange(448,torch::kInt64)*331).remainder(65536)-32768).to(torch::kInt16).reshape({1,8,7,8});
            auto other=values.flip({3}).contiguous();auto device=torch::Device(torch::kCUDA,0);
            same(uf::int16::add({values.to(device),other.to(device)}),uf::int16::add({values,other}),"CUDA add");
            auto scale=torch::tensor({256,512,1024,2048,128,257,700,32767},torch::kInt16).reshape({1,8,1,1});
            same(uf::int16::multiply(values.to(device),scale.to(device)),uf::int16::multiply(values,scale),"CUDA multiply");
            auto lut=torch::arange(-32768,32768,torch::kInt64).to(torch::kInt16);
            same(uf::int16::lookup(values.to(device),lut.to(device)),uf::int16::lookup(values,lut),"CUDA lookup");
            same(uf::int16::wsilu4(values.to(device),lut.to(device)),uf::int16::wsilu4(values,lut),"CUDA WSiLU4");
            auto bytes_tensor=torch::arange(256,torch::kInt64).to(torch::kUInt8);
            same(uf::int16::from_bytes(bytes_tensor.to(device)),uf::int16::from_bytes(bytes_tensor),"CUDA from bytes");
        });
    }
    for (const auto& config:std::vector<std::vector<int>>{{8,6,3,1,1,1},{16,16,1,1,0,1},{8,8,3,2,1,8},{8,12,3,1,1,2}}) {
        const auto label="convolution ci="+std::to_string(config[0])+" groups="+std::to_string(config[5]);
        suite.run(label,[&,config]{
            const int ci=config[0],co=config[1],k=config[2],stride=config[3],pad=config[4],groups=config[5];
            auto x=((torch::arange(ci*56,torch::kInt64)*29).remainder(2048)-1024).to(torch::kInt16).reshape({1,ci,7,8});
            auto w=((torch::arange(co*(ci/groups)*k*k,torch::kInt64)*37).remainder(4096)-2048).to(torch::kInt16).reshape({co,ci/groups,k,k});
            auto b=(torch::arange(co,torch::kInt64)*71-200).to(torch::kInt16);auto expected=convolution_oracle(x,w,b,stride,pad,groups);
            same(uf::int16::conv2d(x,w,b,stride,stride,pad,pad,groups),expected,"CPU convolution");
            if(cuda){auto d=torch::Device(torch::kCUDA,0);same(uf::int16::conv2d(x.to(d),w.to(d),b.to(d),stride,stride,pad,pad,groups),expected,"CUDA convolution");}
        });
    }
#ifdef UF_TEST_CUDA
    if(cuda) for(int kernel:{1,3}) for(int sign:{-1,1}) {
        const auto label="INT64 accumulator and CUDA variants kernel="+std::to_string(kernel)+" sign="+std::to_string(sign);
        suite.run(label,[&,kernel,sign]{
            const auto device=torch::Device(torch::kCUDA,0);
            auto x=torch::full({1,32,5,5},32767,torch::kInt16);
            auto w=torch::full({32,32,kernel,kernel},sign==1?32767:-32768,torch::kInt16);
            auto b=torch::zeros({32},torch::kInt16);
            const int padding=kernel/2;auto expected=convolution_oracle(x,w,b,1,padding,1);
            auto dx=x.to(device),dw=w.to(device),db=b.to(device);
            same(uf::int16::conv2d(x,w,b,1,1,padding,padding,1),expected,"CPU large accumulator");
            same(uf::int16::conv2d(dx,dw,db,1,1,padding,padding,1),expected,"CUDA automatic variant");
            same(uf_conv_generic(dx,dw,c10::optional<torch::Tensor>(db),1,1,padding,padding,1),expected,"CUDA generic variant");
            same(uf_conv_baseline(dx,dw,c10::optional<torch::Tensor>(db),1,1,padding,padding,1),expected,"CUDA baseline variant");
        });
    }
#endif
#ifdef UF_TEST_TRAINING
    const auto image_root=temp/"dataset-images",video_root=temp/"dataset-video";
    const Json image_names=Json::array({"пример.png","two.png"});
    Json frame_names=Json::array();for(int i=0;i<5;++i)frame_names.push_back(std::to_string(i)+".png");
    const Json video_description={{"frames",frame_names},{"seqs",Json::array({{{"path","seq"},{"width",8},{"height",6},{"seq_length",5}}})}};
    suite.run("training synthetic dataset fixture files",[&]{
        std::filesystem::create_directories(image_root);std::filesystem::create_directories(video_root/"seq");
        auto write=[&](const std::filesystem::path& path,int phase){uf::ImagePixels image{8,6,std::vector<std::uint8_t>(144)};
            for(int i=0;i<144;++i)image.rgb[i]=(i*13+phase*17)%256;uf::write_image_png(path,image);};
        for(int i=0;i<2;++i)write(image_root/image_names.at(i).get<std::string>(),i);
        for(int i=0;i<5;++i)write(video_root/"seq"/frame_names.at(i).get<std::string>(),i);
        uf::write_python_json_file_atomic(image_root/"description.json",image_names);
        uf::write_python_json_file_atomic(video_root/"description.json",video_description);
    });
    for(const auto& row:fixtures.at("image_dataset"))suite.run("image dataset Python crop/pad/index seed="+std::to_string(row.at("seed").get<int>()),[&,row]{
        uf::training::ImageFolder dataset(image_root,row.at("height").get<int>(),row.at("width").get<int>(),3,{0.1,0.2,0.4});
        uf::pareto_detail::PythonRandom rng(row.at("seed").get<std::int64_t>());std::vector<std::uint64_t> bounds;
        auto sample=dataset.get(row.at("index").get<int>(),[&](std::uint64_t upper){bounds.push_back(upper);return rng.below(upper);});
        same_sample(sample,row.at("sample"));require(bounds==row.at("bounds").get<std::vector<std::uint64_t>>(),"Image draw order/bounds");
        require(dataset.size()==2,"Image dataset length");
    });
    for(const auto& row:fixtures.at("video_dataset"))suite.run("video dataset Python windows/bounce/groups count="+std::to_string(row.at("count").get<int>()),[&,row]{
        uf::training::VideoFolder dataset(video_root,row.at("height").get<int>(),row.at("width").get<int>(),3,{0.1,0.2,0.4},row.at("count").get<int>(),row.at("group").get<int>());
        uf::pareto_detail::PythonRandom rng(row.at("seed").get<std::int64_t>());std::vector<std::uint64_t> bounds;
        auto sample=dataset.get(0,[&](std::uint64_t upper){bounds.push_back(upper);return rng.below(upper);});
        same_sample(sample,row.at("sample"));require(bounds==row.at("bounds").get<std::vector<std::uint64_t>>(),"Video draw order/bounds");
        require(dataset.size()==1 && dataset.frame_count()==row.at("count").get<int>(),"Video dataset settings");
    });
    suite.run("dataset snapshots settings and validation",[&]{
        uf::training::ImageFolder image(image_root,4,6,3,{0.1,0.2,0.4});image.set_patch_size(10,8);
        auto image_state=image.worker_snapshot();image_state["root_bytes"]=std::vector<std::uint8_t>{'/','n','o','n','e'};
        uf::training::ImageFolder restored(image_state);require(restored.size()==2 && restored.patch_size()==std::make_pair(std::int64_t(10),std::int64_t(8)),"Image snapshot/settings");
        uf::training::VideoFolder video(video_root,4,6,3,{0.1,0.2,0.4});video.set_patch_size(10,8);video.set_frame_count(8);
        auto state=video.worker_snapshot();state["root_bytes"]=std::vector<std::uint8_t>{'/','n','o','n','e'};
        uf::training::VideoFolder copy(state);require(copy.frame_count()==8 && copy.patch_size()==std::make_pair(std::int64_t(10),std::int64_t(8)),"Video snapshot/settings");
        throws([&]{image.set_patch_size(-1,4);});throws([&]{video.set_frame_count(0);});
        throws([&]{uf::training::ImageFolder invalid(image_root,4,6,3,{0.1});});
        throws([&]{image.get(2,[](std::uint64_t){return 0;});});throws([&]{image.get(-3,[](std::uint64_t){return 0;});});
        throws([&]{image.get(0,[](std::uint64_t upper){return upper;});});
    });
    suite.run("dataset collation preserves fields and scalar dtypes",[&]{
        uf::training::ImageFolder image(image_root,4,6,3,{0.1,0.2,0.4});
        uf::pareto_detail::PythonRandom a(0),b(5);auto first=image.get(0,[&](std::uint64_t n){return a.below(n);});
        auto second=image.get(-1,[&](std::uint64_t n){return b.below(n);});
        auto batch=uf::training::collate_samples({first,second});require(batch.size()==3,"Batch field count");
        for(int i=0;i<3;++i)same(batch[i],torch::stack({first[i],second[i]}),"Collated field");
        require(batch[1].scalar_type()==torch::kInt32 && batch[2].scalar_type()==torch::kFloat32,"Scalar field dtypes");
        throws([&]{uf::training::collate_samples({});});throws([&]{uf::training::collate_samples({first,{second[0]}});});
    });
    for(const auto& row:fixtures.at("training_sampler"))suite.run("training sampler and worker RNG Python reference rank="+std::to_string(row.at("rank").get<int>()),[&,row]{
        torch::manual_seed(row.at("seed").get<std::int64_t>());uf::training::TrainingLoaderOptions options;
        options.batch_size=row.at("batch").get<int>();options.workers=2;options.prefetch_factor=2;options.pin_memory=false;
        options.rank=row.at("rank").get<int>();options.world_size=row.at("world").get<int>();
        uf::training::TrainingDataLoader loader(7,[](std::int64_t index,const uf::training::RandomBelow& random){
            return uf::training::DatasetSample{torch::tensor(index,torch::kInt64),torch::tensor(static_cast<std::int32_t>(random(1000)),torch::kInt32)};},options);
        loader.reset(row.at("epoch").get<int>());require(loader.batch_count()==row.at("batches").size(),"Loader drop-last count");
        for(const auto& expected:row.at("batches")) {auto batch=loader.next();require(bool(batch),"Missing ordered batch");
            same(batch->at(0),torch::tensor(expected.at("indices").get<std::vector<std::int64_t>>(),torch::kInt64),"Sampler order");
            same(batch->at(1),torch::tensor(expected.at("draws").get<std::vector<std::int32_t>>(),torch::kInt32),"Worker private RNG");}
        require(!loader.next() && !loader.next(),"Loader EOF");
    });
    suite.run("training loader batch error propagation and epoch reset",[&]{
        uf::training::TrainingLoaderOptions options;options.batch_size=2;options.workers=1;options.pin_memory=false;bool fail=true;
        uf::training::TrainingDataLoader loader(4,[&](std::int64_t index,const uf::training::RandomBelow&){
            if(fail){fail=false;throw std::runtime_error("synthetic dataset failure");}
            return uf::training::DatasetSample{torch::tensor(index,torch::kInt64)};},options);
        loader.reset(0);bool caught=false;try{loader.next();}catch(const std::runtime_error& error){caught=std::string(error.what())=="synthetic dataset failure";}
        require(caught && bool(loader.next()) && !loader.next(),"Batch error/continuation/EOF");
        loader.reset(1);require(bool(loader.next()) && bool(loader.next()) && !loader.next(),"Reset after failed epoch");
    });
    suite.run("training loader option validation and empty distributed rank",[&]{
        auto reader=[](std::int64_t,const uf::training::RandomBelow&){return uf::training::DatasetSample{torch::zeros({1})};};
        uf::training::TrainingLoaderOptions options;options.batch_size=2;options.workers=1;options.pin_memory=false;
        throws([&]{uf::training::TrainingDataLoader loader(0,reader,options);});
        auto invalid=options;invalid.workers=0;throws([&]{uf::training::TrainingDataLoader loader(4,reader,invalid);});
        invalid=options;invalid.rank=0;invalid.world_size=3;throws([&]{uf::training::TrainingDataLoader loader(4,reader,invalid);});
        options.rank=0;options.world_size=2;uf::training::TrainingDataLoader empty(0,reader,options);
        empty.reset(0);require(empty.batch_count()==0 && !empty.next(),"Empty distributed rank");
    });
    suite.run("fresh process image loader Python sample parity and completion",[&]{
        auto before=child_processes();uf::training::ImageFolder image(image_root,4,6,3,{0.1,0.2,0.4});
        uf::training::TrainingLoaderOptions options;options.batch_size=1;options.workers=1;options.pin_memory=false;
        {torch::manual_seed(2026);uf::training::TrainingDataLoader loader(image,options);loader.reset(0);
         for(const auto& expected:fixtures.at("process_image_loader")){auto batch=loader.next();require(bool(batch),"Missing process batch");same_sample(*batch,expected);}
         require(!loader.next(),"Process loader EOF");}
        require(child_processes()==before,"Completed loader child not reaped");
    });
    suite.run("fresh process loader early cancellation reaps child",[&]{
        auto before=child_processes();uf::training::ImageFolder image(image_root,4,6,3,{0.1,0.2,0.4});
        uf::training::TrainingLoaderOptions options;options.batch_size=1;options.workers=1;options.pin_memory=false;
        {uf::training::TrainingDataLoader loader(image,options);loader.reset(0);require(bool(loader.next()),"Process child did not produce batch");
         require(child_processes().size()==before.size()+1,"Expected one live loader child");}
        require(child_processes()==before,"Canceled loader child not reaped");
    });
    suite.run("fresh process dataset errors survive transport and cleanup",[&]{
        auto bad_root=temp/"missing-dataset";std::filesystem::create_directories(bad_root);
        uf::write_python_json_file_atomic(bad_root/"description.json",Json::array({"missing.png","missing.png"}));
        auto before=child_processes();uf::training::ImageFolder image(bad_root,4,6,3,{0.1,0.2,0.4});
        uf::training::TrainingLoaderOptions options;options.batch_size=1;options.workers=1;options.pin_memory=false;
        {uf::training::TrainingDataLoader loader(image,options);loader.reset(0);throws([&]{loader.next();});throws([&]{loader.next();});require(!loader.next(),"Errored process loader EOF");}
        require(child_processes()==before,"Errored loader child not reaped");
    });
    if(cuda) {
        const auto device=torch::Device(torch::kCUDA,0);
        const auto transfer=[&](const uf::training::DatasetSample& batch){
            uf::training::DatasetSample copied;
            for(const auto& tensor:batch){
                require(tensor.device().is_cpu() && tensor.is_pinned(),"Loader field is not pinned CPU memory");
                copied.push_back(tensor.to(device,tensor.scalar_type(),true,false));
            }
            torch::cuda::synchronize();
            for(std::size_t i=0;i<batch.size();++i)same(copied[i],batch[i],"Pinned asynchronous CUDA handoff");
            return copied;
        };
        suite.run("pinned callback loader Python sampler parity and CUDA handoff",[&]{
            const auto& row=fixtures.at("training_sampler").at(0);torch::manual_seed(row.at("seed").get<std::int64_t>());
            uf::training::TrainingLoaderOptions options;options.batch_size=2;options.workers=2;options.device=device;options.pin_memory=true;
            uf::training::TrainingDataLoader loader(7,[](std::int64_t index,const uf::training::RandomBelow& random){
                return uf::training::DatasetSample{torch::tensor(index,torch::kInt64),torch::tensor(static_cast<std::int32_t>(random(1000)),torch::kInt32)};},options);
            loader.reset(0);uf::training::DatasetSample retained;
            for(const auto& expected:row.at("batches")){auto batch=loader.next();require(bool(batch),"Missing pinned callback batch");
                same(batch->at(0),torch::tensor(expected.at("indices").get<std::vector<std::int64_t>>(),torch::kInt64),"Pinned sampler order");
                same(batch->at(1),torch::tensor(expected.at("draws").get<std::vector<std::int32_t>>(),torch::kInt32),"Pinned worker RNG");
                auto copied=transfer(*batch);if(retained.empty())retained=std::move(copied);}
            require(!loader.next(),"Pinned callback EOF");
            same(retained[0],torch::tensor(row.at("batches").at(0).at("indices").get<std::vector<std::int64_t>>(),torch::kInt64),"Retained CUDA batch after shutdown");
        });
        suite.run("pinned fresh-process image batches Python parity and lifetime",[&]{
            auto before=child_processes();uf::training::ImageFolder image(image_root,4,6,3,{0.1,0.2,0.4});
            uf::training::TrainingLoaderOptions options;options.batch_size=1;options.workers=1;options.device=device;options.pin_memory=true;
            uf::training::DatasetSample retained,retained_cuda;
            {torch::manual_seed(2026);uf::training::TrainingDataLoader loader(image,options);loader.reset(0);
             for(const auto& expected:fixtures.at("process_image_loader")){auto batch=loader.next();require(bool(batch),"Missing pinned process batch");same_sample(*batch,expected);
                auto copied=transfer(*batch);same_sample(copied,expected);if(retained.empty()){retained=*batch;retained_cuda=std::move(copied);}}
             require(!loader.next(),"Pinned process EOF");}
            same_sample(retained,fixtures.at("process_image_loader").at(0));same_sample(retained_cuda,fixtures.at("process_image_loader").at(0));
            require(child_processes()==before,"Pinned completed child not reaped");
        });
        suite.run("pinning thread propagates batch errors and supports reset",[&]{
            uf::training::TrainingLoaderOptions options;options.batch_size=2;options.workers=1;options.device=device;options.pin_memory=true;bool fail=true;
            uf::training::TrainingDataLoader loader(4,[&](std::int64_t index,const uf::training::RandomBelow&){
                if(fail){fail=false;throw std::runtime_error("synthetic pinning input error");}
                return uf::training::DatasetSample{torch::tensor(index,torch::kInt64)};},options);
            loader.reset(0);bool caught=false;try{loader.next();}catch(const std::runtime_error& error){caught=std::string(error.what())=="synthetic pinning input error";}
            require(caught,"Pinned pipeline lost dataset error");auto batch=loader.next();require(bool(batch),"Pinned continuation batch missing");transfer(*batch);
            require(!loader.next(),"Pinned failed epoch EOF");loader.reset(1);
            for(int i=0;i<2;++i){batch=loader.next();require(bool(batch),"Pinned reset batch missing");transfer(*batch);}require(!loader.next(),"Pinned reset EOF");
        });
        suite.run("pinned fresh-process early close retains data and reaps child",[&]{
            auto before=child_processes();uf::training::ImageFolder image(image_root,4,6,3,{0.1,0.2,0.4});
            uf::training::TrainingLoaderOptions options;options.batch_size=1;options.workers=1;options.device=device;options.pin_memory=true;
            uf::training::DatasetSample retained;std::vector<torch::Tensor> snapshots;
            {uf::training::TrainingDataLoader loader(image,options);loader.reset(0);auto batch=loader.next();require(bool(batch),"Pinned child produced no batch");
             retained=*batch;for(const auto& tensor:retained)snapshots.push_back(tensor.clone());transfer(retained);
             require(child_processes().size()==before.size()+1,"Expected one pinned loader child");}
            require(child_processes()==before,"Pinned canceled child not reaped");
            for(std::size_t i=0;i<retained.size();++i)same(retained[i],snapshots[i],"Pinned data lifetime after cancellation");transfer(retained);
        });
        suite.run("pinned fresh-process dataset errors reach EOF and reap child",[&]{
            auto before=child_processes();uf::training::ImageFolder image(temp/"missing-dataset",4,6,3,{0.1,0.2,0.4});
            uf::training::TrainingLoaderOptions options;options.batch_size=1;options.workers=1;options.device=device;options.pin_memory=true;
            {uf::training::TrainingDataLoader loader(image,options);loader.reset(0);throws([&]{loader.next();});throws([&]{loader.next();});require(!loader.next(),"Pinned error EOF");}
            require(child_processes()==before,"Pinned error child not reaped");
        });
    }
    for(bool nested:{false,true}) for(bool gpu:{false,true}) {
        if(gpu && !cuda)continue;
        suite.run("activation recomputation Python parity nested="+std::to_string(nested)+" cuda="+std::to_string(gpu),[&,nested,gpu]{
            auto device=gpu?torch::Device(torch::kCUDA,0):torch::Device(torch::kCPU);
            auto x=torch::tensor({-0.75,0.25,1.5},torch::kFloat32).to(device).set_requires_grad(true);
            auto weight=torch::tensor({0.5,-1.0,2.0},torch::kFloat32).to(device).set_requires_grad(true);
            int outer_calls=0,inner_calls=0;
            auto output=uf::training::checkpoint_activations([&](const uf::training::CheckpointInputs& inputs){
                ++outer_calls;torch::Tensor product;
                if(nested)product=uf::training::checkpoint_tensor([&](const uf::training::CheckpointInputs& inner){
                    ++inner_calls;return inner[0]*inner[1];},inputs);
                else product=inputs[0]*inputs[1];
                return uf::training::CheckpointInputs{product.sin(),product.square()};
            },{x,weight});
            const auto& reference=fixtures.at("activation_recompute").at(nested?1:0);
            for(std::size_t i=0;i<output.size();++i)
                require(torch::allclose(output[i].detach().cpu(),torch::tensor(reference.at("outputs").at(i).get<std::vector<float>>()),1e-6,1e-6),"Checkpoint forward value");
            auto loss=(output[0]+0.125*output[1]).sum();
            for(const auto& name:{"first","second"}) {
                loss.backward({},std::string(name)=="first");
                const auto& expected=reference.at(name);
                require(torch::allclose(x.grad().cpu(),torch::tensor(expected.at("x_gradient").get<std::vector<float>>()),1e-6,1e-6),"Checkpoint input gradient");
                require(torch::allclose(weight.grad().cpu(),torch::tensor(expected.at("weight_gradient").get<std::vector<float>>()),1e-6,1e-6),"Checkpoint parameter gradient");
                require(outer_calls==expected.at("calls").at("outer").get<int>() && inner_calls==expected.at("calls").at("inner").get<int>(),"Checkpoint replay counts");
                x.mutable_grad()=torch::Tensor{};weight.mutable_grad()=torch::Tensor{};
            }
        });
    }
    suite.run("activation recomputation no-grad and callback validation",[&]{
        throws([&]{uf::training::checkpoint_activations({},{});});
        auto x=torch::tensor({1.0,2.0},torch::TensorOptions().requires_grad(true));int calls=0;
        const torch::NoGradGuard no_grad;
        auto output=uf::training::checkpoint_tensor([&](const uf::training::CheckpointInputs& inputs){++calls;return inputs[0].square();},{x});
        require(calls==1 && !output.requires_grad(),"No-grad checkpoint policy");
        same(output,torch::tensor({1.0,4.0}),"No-grad checkpoint output");
    });
    suite.run("activation recomputation input version rejection",[&]{
        auto x=torch::tensor({1.0,2.0},torch::TensorOptions().requires_grad(true));
        auto output=uf::training::checkpoint_tensor([](const uf::training::CheckpointInputs& inputs){return inputs[0].square();},{x});
        {const torch::NoGradGuard no_grad;x.add_(1);}
        throws([&]{output.sum().backward();});
    });
    suite.run("activation recomputation metadata mismatch rejection",[&]{
        auto x=torch::tensor({1.0,2.0},torch::TensorOptions().requires_grad(true));bool replay=false;
        auto output=uf::training::checkpoint_tensor([&](const uf::training::CheckpointInputs& inputs){
            auto value=replay?inputs[0].slice(0,0,1):inputs[0];return value.square();},{x});
        replay=true;bool caught=false;
        try{output.sum().backward();}catch(const uf::training::ActivationCheckpointError& error){
            caught=std::string(error.what()).find("different shape, dtype or device")!=std::string::npos;}
        require(caught,"Recompute metadata error missing");
    });
    suite.run("activation recomputation preserves no RNG state",[&]{
        torch::manual_seed(314159);int calls=0;
        auto x=torch::tensor({1.0,2.0,3.0},torch::TensorOptions().requires_grad(true));
        auto output=uf::training::checkpoint_tensor([&](const uf::training::CheckpointInputs& inputs){
            ++calls;return inputs[0]*torch::rand_like(inputs[0]);},{x});
        output.sum().backward();auto next=torch::rand({3});const auto& expected=fixtures.at("recompute_rng");
        same(output.detach(),torch::tensor(expected.at("output").get<std::vector<float>>()),"Stochastic forward");
        same(x.grad(),torch::tensor(expected.at("gradient").get<std::vector<float>>()),"Fresh replay RNG gradient");
        same(next,torch::tensor(expected.at("next_draw").get<std::vector<float>>()),"RNG advancement");
        require(calls==expected.at("calls").get<int>(),"Stochastic replay count");
    });
    for(bool gpu:{false,true}) {
        if(gpu && !cuda)continue;
        suite.run("activation recomputation higher-order Python gradients cuda="+std::to_string(gpu),[&,gpu]{
            auto device=gpu?torch::Device(torch::kCUDA,0):torch::Device(torch::kCPU);
            auto x=torch::tensor({-0.75,0.25,1.5},torch::kFloat32).to(device).set_requires_grad(true);
            auto output=uf::training::checkpoint_tensor([](const uf::training::CheckpointInputs& inputs){return inputs[0].sin().square();},{x});
            auto first=torch::autograd::grad({output.sum()},{x},{},true,true).at(0);
            auto second=torch::autograd::grad({first.sum()},{x}).at(0);
            const auto& expected=fixtures.at("recompute_higher_order");
            require(torch::allclose(first.detach().cpu(),torch::tensor(expected.at("first").get<std::vector<float>>()),1e-6,1e-6),"First functional gradient");
            require(torch::allclose(second.cpu(),torch::tensor(expected.at("second").get<std::vector<float>>()),1e-6,1e-6),"Second functional gradient");
        });
        suite.run("activation recomputation autocast capture and restore cuda="+std::to_string(gpu),[&,gpu]{
            auto device=gpu?torch::Device(torch::kCUDA,0):torch::Device(torch::kCPU);
            const auto type=gpu?at::kCUDA:at::kCPU;
            const auto dtype=gpu?at::kHalf:at::kBFloat16;
            const auto& expected=fixtures.at("recompute_autocast").at(gpu?1:0);
            auto x=((torch::arange(8,torch::kFloat32).reshape({2,4})-3)/8).to(device).set_requires_grad(true);
            auto weight=((torch::arange(12,torch::kFloat32).reshape({4,3})-5)/16).to(device).set_requires_grad(true);
            std::vector<bool> enabled,cache;std::vector<at::ScalarType> dtypes;torch::Tensor output;
            {const TestAutocastScope scope(type,dtype,true,true);
             output=uf::training::checkpoint_tensor([&](const uf::training::CheckpointInputs& inputs){
                enabled.push_back(at::autocast::is_autocast_enabled(type));dtypes.push_back(at::autocast::get_autocast_dtype(type));
                cache.push_back(at::autocast::is_autocast_cache_enabled());return torch::matmul(inputs[0],inputs[1]).square();},{x,weight});}
            const auto expected_dtype=expected.at("output_dtype").get<std::string>();
            require((output.scalar_type()==at::kFloat && expected_dtype=="torch.float32") ||
                (output.scalar_type()==at::kHalf && expected_dtype=="torch.float16") ||
                (output.scalar_type()==at::kBFloat16 && expected_dtype=="torch.bfloat16"),"Autocast forward output dtype");
            {const TestAutocastScope caller(type,at::kFloat,false,false);output.sum().backward();
             require(!at::autocast::is_autocast_enabled(type) && at::autocast::get_autocast_dtype(type)==at::kFloat &&
                !at::autocast::is_autocast_cache_enabled(),"Recompute changed caller autocast state");}
            require(enabled.size()==expected.at("seen").size(),"Autocast replay count");
            for(std::size_t i=0;i<enabled.size();++i)
                require(enabled[i]==expected.at("seen").at(i).at("enabled").get<bool>() && dtypes[i]==dtype &&
                    cache[i]==expected.at("seen").at(i).at("cache").get<bool>(),"Forward/replay autocast context");
            require(torch::allclose(output.detach().cpu().to(torch::kFloat32).flatten(),torch::tensor(expected.at("output").get<std::vector<float>>()),1e-6,1e-6),"Autocast forward values");
            require(torch::allclose(x.grad().cpu().flatten(),torch::tensor(expected.at("x_gradient").get<std::vector<float>>()),1e-6,1e-6),"Autocast input gradient");
            require(torch::allclose(weight.grad().cpu().flatten(),torch::tensor(expected.at("weight_gradient").get<std::vector<float>>()),1e-6,1e-6),"Autocast weight gradient");
        });
    }
    suite.run("primitive quantization and lower-bound gradients",[&]{
        auto x=torch::tensor({-0.5,0.5,1.5},torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
        uf::training::quantize(x).sum().backward();same(x.grad(),torch::ones_like(x),"Quantization gradient");
        x=torch::tensor({-1.0,-1.0,1.0},torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
        auto output=uf::training::lower_bound(x,0.0);output.backward(torch::tensor({1.0,-1.0,1.0}));
        same(x.grad(),torch::tensor({0.0,-1.0,1.0}),"Lower-bound gradient");
    });
    for(bool gpu:{false,true}) {
        if(gpu && !cuda)continue;
        auto device=gpu?torch::Device(torch::kCUDA,0):torch::Device(torch::kCPU);
        suite.run("weighted YUV/RGB loss Python values and gradients cuda="+std::to_string(gpu),[&,device]{
            auto x=(torch::arange(144,torch::kFloat32).reshape({2,3,4,6}).remainder(29)/32).to(device).set_requires_grad(true);
            auto reconstruction=(x.detach().flip({3})*0.95+0.02).set_requires_grad(true);
            auto components=uf::training::mse_yuv_rgb(x,reconstruction);
            auto weighted=uf::training::mse_weighted_average(components.first,components.second,24);
            weighted.sum().backward();const auto& expected=fixtures.at("weighted_mse");
            close_float(components.first,expected.at("yuv"),"YUV loss");close_float(components.second,expected.at("rgb"),"RGB loss");
            close_float(weighted,expected.at("weighted"),"Weighted loss");
            close_float(x.grad(),expected.at("input_gradient"),"Loss input gradient");
            close_float(reconstruction.grad(),expected.at("reconstruction_gradient"),"Loss reconstruction gradient");
        });
        suite.run("color conversion pooling and frame loss Python parity cuda="+std::to_string(gpu),[&,device]{
            auto input=((torch::arange(24,torch::kFloat32)-5).reshape({1,3,2,4})/16).to(device);
            for(const auto& row:fixtures.at("color_transforms")) {
                const bool bgr=row.at("bgr").get<bool>();
                close_float(uf::training::rgb_to_ycbcr(input,bgr),row.at("yuv"),"RGB to YUV");
                close_float(uf::training::ycbcr_to_rgb(input,bgr,true),row.at("rgb_clamped"),"Clamped YUV to RGB");
                close_float(uf::training::ycbcr_to_rgb(input,bgr,false),row.at("rgb_unclamped"),"Unclamped YUV to RGB");
            }
            auto yuv=(torch::arange(144,torch::kFloat32).reshape({2,3,4,6}).remainder(29)/32).to(device);
            auto planes=uf::training::yuv444_to_420(yuv);
            close_float(planes.first,fixtures.at("color_pool").at("y"),"Luma slicing");
            close_float(planes.second,fixtures.at("color_pool").at("uv"),"Chroma pooling");
            std::vector<torch::Tensor> frames;for(int i=0;i<9;++i)frames.push_back(torch::tensor({i+1.0}).to(device));
            require(uf::training::mse_eight_frame_sum(frames,{1,0.5,0.25,99}).item<double>()==fixtures.at("eight_frame_mse").get<double>(),"Eight-frame weighted sum");
            throws([&]{uf::training::mse_eight_frame_sum({frames[0]},{1,0.5,0.25});});
        });
        suite.run("entropy probability Python values and gradients cuda="+std::to_string(gpu),[&,device]{
            auto x=((torch::arange(24,torch::kFloat32)-12)/8).reshape({1,2,3,4}).to(device).set_requires_grad(true);
            auto h=((torch::arange(8,torch::kFloat32)-4)/16).reshape({1,2,4}).to(device).set_requires_grad(true);
            auto b=((torch::arange(8,torch::kFloat32)-3)/32).reshape({1,2,4}).to(device).set_requires_grad(true);
            auto a=((torch::arange(6,torch::kFloat32)-2)/16).reshape({1,2,3}).to(device).set_requires_grad(true);
            auto probability=uf::training::bit_estimator_z_probability(x,h,b,a),mass=uf::training::bit_estimator_z_mass(x,h,b,a);
            (probability.sum()+mass.sum()).backward();const auto& expected=fixtures.at("bit_probability");
            close_float(probability,expected.at("probability"),"Accumulated entropy probability");close_float(mass,expected.at("mass"),"Entropy mass");
            int index=0;for(const auto& tensor:{x,h,b,a})close_float(tensor.grad(),expected.at("gradients").at(index++),"Entropy parameter gradient");
        });
    }
    for(bool foreach:{false,true}) for(bool gpu:{false,true}) {
        if(gpu && !cuda)continue;
        suite.run("AdamW primitive Python parity foreach="+std::to_string(foreach)+" cuda="+std::to_string(gpu),[&,foreach,gpu]{
            auto device=torch::Device(gpu?torch::kCUDA:torch::kCPU,0);
            if(!gpu)device=torch::Device(torch::kCPU);
            auto p=torch::tensor({1.0,-2.0,0.5},torch::kFloat32).to(device).set_requires_grad(true);
            uf::training::AdamWOptions options(0.01);options.betas(std::make_tuple(0.8,0.95));
            options.weight_decay(0.1);options.amsgrad(true);options.foreach(foreach);
            uf::training::AdamW optimizer({p},options);
            for(const auto& row:fixtures.at("adamw")){
                p.mutable_grad()=torch::tensor(row.at("gradient").get<std::vector<float>>(),torch::kFloat32).to(device);
                optimizer.step();
                auto expected=torch::tensor(row.at("parameter").get<std::vector<float>>(),torch::kFloat32);
                require(torch::allclose(p.detach().cpu(),expected,1e-6,1e-6),"AdamW parameter update");
                auto& state=dynamic_cast<uf::training::AdamWParamState&>(*optimizer.state().at(p.unsafeGetTensorImpl()));
                require(torch::allclose(state.exp_avg().cpu(),torch::tensor(row.at("exp_avg").get<std::vector<float>>()),1e-6,1e-7),"AdamW first moment");
                require(torch::allclose(state.exp_avg_sq().cpu(),torch::tensor(row.at("exp_avg_sq").get<std::vector<float>>()),1e-6,1e-7),"AdamW second moment");
                require(state.step().item<double>()==row.at("step").get<double>(),"AdamW step counter");
            }
        });
    }
    for(bool scaled:{false,true}) for(bool gpu:{false,true}) {
        if(gpu && !cuda)continue;
        suite.run("AdamW fused Python parity scaled="+std::to_string(scaled)+" cuda="+std::to_string(gpu),[&,scaled,gpu]{
            auto device=gpu?torch::Device(torch::kCUDA,0):torch::Device(torch::kCPU);
            auto p=torch::tensor({1.0,-2.0,0.5},torch::kFloat32).to(device).set_requires_grad(true);
            uf::training::AdamWOptions options(0.01);options.betas(std::make_tuple(0.8,0.95));
            options.weight_decay(0.1);options.amsgrad(true);options.fused(true);
            uf::training::AdamW optimizer({p},options);
            for(const auto& row:fixtures.at(scaled?"adamw_fused_amp":"adamw")) {
                p.mutable_grad()=torch::tensor(row.at("gradient").get<std::vector<float>>()).to(device)*(scaled?8:1);
                if(scaled)optimizer.amp_scaling(torch::tensor(8.0),torch::tensor(row.at("overflow").get<bool>()?1.0:0.0));
                optimizer.step();
                auto& state=dynamic_cast<uf::training::AdamWParamState&>(*optimizer.state().at(p.unsafeGetTensorImpl()));
                require(torch::allclose(p.detach().cpu(),torch::tensor(row.at("parameter").get<std::vector<float>>()),1e-6,1e-6),"Fused AdamW parameter");
                require(torch::allclose(state.exp_avg().cpu(),torch::tensor(row.at("exp_avg").get<std::vector<float>>()),1e-6,1e-7),"Fused AdamW first moment");
                require(torch::allclose(state.exp_avg_sq().cpu(),torch::tensor(row.at("exp_avg_sq").get<std::vector<float>>()),1e-6,1e-7),"Fused AdamW second moment");
                require(state.step().item<double>()==row.at("step").get<double>(),"Fused AdamW step/overflow counter");
                if(scaled)require(torch::allclose(state.max_exp_avg_sq().cpu(),torch::tensor(row.at("max_exp_avg_sq").get<std::vector<float>>()),1e-6,1e-7),"Fused AMSGrad maximum");
            }
        });
    }
    suite.run("AdamW invalid options and nonfused AMP rejection",[&]{
        auto p=torch::tensor({1.0},torch::TensorOptions().requires_grad(true));
        throws([&]{uf::training::AdamW optimizer({p},uf::training::AdamWOptions(-0.01));});
        uf::training::AdamWOptions invalid;invalid.betas(std::make_tuple(1.0,0.9));
        throws([&]{uf::training::AdamW optimizer({p},invalid);});
        invalid=uf::training::AdamWOptions{};invalid.fused(true);invalid.foreach(true);
        throws([&]{uf::training::AdamW optimizer({p},invalid);});
        for(bool foreach:{false,true}) {
            uf::training::AdamWOptions options;options.foreach(foreach);uf::training::AdamW optimizer({p},options);
            p.mutable_grad()=torch::ones_like(p);optimizer.amp_scaling(torch::tensor(8.0),torch::tensor(0.0));
            throws([&]{optimizer.step();});same(p.detach(),torch::tensor({1.0}),"Rejected AMP parameter");
        }
    });
#endif
#ifdef UF_TEST_EVALUATION
    suite.run("PSNR Python parity across physical layouts",[&]{
        auto pair=metric_pair(12,5,3);
        std::vector<std::pair<torch::Tensor,torch::Tensor>> cases={pair,{pair.first.permute({1,0,2}),pair.second.permute({1,0,2})},
            {pair.first.slice(2,0,5,2),pair.second.slice(2,0,5,2)}};
        for(std::size_t i=0;i<cases.size();++i)
            require(std::abs(uf::evaluation::calc_psnr(cases[i].first,cases[i].second)-fixtures.at("metric_psnr").at(i).get<double>())<1e-12,"PSNR layout parity");
    });
    suite.run("PSNR cap and nonfinite sentinel",[&]{
        auto zero=torch::zeros({3},torch::kFloat64);
        require(uf::evaluation::calc_psnr(zero,zero)==99.9 && uf::evaluation::calc_psnr(zero,zero+1e-6)==99.9,"PSNR cap/threshold");
        for(double value:{std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()})
            require(uf::evaluation::calc_psnr(zero,zero+value)==-999.9,"Nonfinite PSNR sentinel");
        require(uf::evaluation::calc_psnr(torch::empty({0},torch::kFloat64),torch::empty({0},torch::kFloat64))==-999.9,"Empty PSNR sentinel");
    });
    for(const auto& row:fixtures.at("metric_gaussian"))suite.run("Gaussian evaluation window size="+std::to_string(row.at("size").get<int>()),[&,row]{
        auto actual=uf::evaluation::fspecial_gauss(row.at("size").get<int>(),row.at("sigma").get<double>());
        auto expected=torch::tensor(row.at("values").get<std::vector<double>>(),torch::kFloat64);
        require(torch::allclose(actual.flatten(),expected,1e-13,1e-15),"Gaussian window Python parity");
    });
    for(const auto& row:fixtures.at("metric_ssim"))suite.run("SSIM Python valid convolution height="+std::to_string(row.at("height").get<int>()),[&,row]{
        auto pair=metric_pair(row.at("height").get<int>(),row.at("width").get<int>());
        auto maps=uf::evaluation::calc_ssim(pair.first,pair.second);
        require(maps.first.sizes().vec()==row.at("shape").get<std::vector<std::int64_t>>(),"SSIM output shape");
        require(torch::allclose(maps.first.flatten(),torch::tensor(row.at("ssim").get<std::vector<double>>(),torch::kFloat64),1e-9,1e-9),"SSIM map Python parity");
        require(torch::allclose(maps.second.flatten(),torch::tensor(row.at("cs").get<std::vector<double>>(),torch::kFloat64),1e-9,1e-9),"Contrast map Python parity");
    });
    for(const auto& row:fixtures.at("metric_msssim"))suite.run("MS-SSIM Python pyramid height="+std::to_string(row.at("height").get<int>()),[&,row]{
        auto pair=metric_pair(row.at("height").get<int>(),row.at("width").get<int>());
        require(std::abs(uf::evaluation::calc_msssim(pair.first,pair.second)-row.at("value").get<double>())<1e-9,"MS-SSIM four/five levels");
    });
    suite.run("RGB MS-SSIM Python channel average",[&]{
        auto pair=metric_pair(88,91,3);require(std::abs(uf::evaluation::calc_msssim_rgb(pair.first,pair.second)-fixtures.at("metric_msssim_rgb").get<double>())<1e-9,"RGB MS-SSIM");
    });
    suite.run("evaluation metric shape and size rejection",[&]{
        auto small=torch::zeros({87,88},torch::kFloat64);throws([&]{uf::evaluation::calc_msssim(small,small);});
        throws([&]{uf::evaluation::calc_ssim(torch::zeros({3,4}),torch::zeros({4,3}));});
        auto mixed=torch::zeros({6,12});throws([&]{uf::evaluation::calc_ssim(mixed,mixed);});
        throws([&]{uf::evaluation::calc_msssim_rgb(torch::zeros({2,88,88}),torch::zeros({2,88,88}));});
    });
    for(int fields:{1,4})suite.run("evaluation report Python totals and prefixes fields="+std::to_string(fields),[&,fields]{
        for(const auto& row:fixtures.at("quality_logs"))if(row.at("fields")==fields) {
            std::vector<uf::evaluation::FrameQuality> quality;
            for(int i=0;i<4;++i)quality.push_back({row.at("psnr").at(i).get<std::vector<double>>(),row.at("ssim").at(i).get<std::vector<double>>()});
            auto actual=uf::evaluation::generate_log_json(3,91,2.5,row.at("types").get<std::vector<int>>(),row.at("bits").get<std::vector<std::uint64_t>>(),quality,true,0.02,0.01);
            require(Json::parse(actual.dump())==row.at("expected"),"Evaluation report fields/UInt128 totals/verbose prefixes");
        }
    });
#endif
#ifdef UF_TEST_FP16
    if(cuda) {
        suite.run("FP16 CUTLASS pointwise bias and fused WSiLU",[&]{
            auto device=torch::Device(torch::kCUDA,0);
            auto x=((torch::arange(512,torch::kFloat32).remainder(32)-16)/32)
                .reshape({1,32,4,4}).to(device,torch::kFloat16).contiguous(torch::MemoryFormat::ChannelsLast);
            auto source_w=((torch::arange(1024,torch::kFloat32).remainder(16)-8)/64).reshape({32,32,1,1});
            auto w=uf::fp16_parameter_to_device(source_w,device);
            auto b=((torch::arange(32,torch::kFloat32)-16)/32).to(device,torch::kFloat16);
            require(w.stride(3)==32,"Pointwise weight channels-last leading stride is "+std::to_string(w.stride(3))+", expected 32");
            auto expected=torch::conv2d(x.cpu().to(torch::kFloat32),w.cpu().to(torch::kFloat32),b.cpu().to(torch::kFloat32)).to(torch::kFloat16);
            auto output=torch::empty_like(x);
            same(conv1x1_bias(87,x,w,b,output),expected,"FP16 pointwise convolution");
            auto activated=(4*expected.to(torch::kFloat32)).sigmoid()*expected.to(torch::kFloat32);
            auto actual=conv1x1_bias_wsilu(87,x,w,b,output).cpu().to(torch::kFloat32);
            require(torch::allclose(actual,activated,0.002,0.002),"FP16 fused WSiLU tolerance");
        });
        suite.run("FP16 pixel shuffle and unshuffle",[&]{
            auto device=torch::Device(torch::kCUDA,0);
            auto x=(torch::arange(3*16*16,torch::kFloat32).remainder(128)/64-1)
                .reshape({1,3,16,16}).to(device,torch::kFloat16).contiguous(torch::MemoryFormat::ChannelsLast);
            auto unshuffled=torch::empty({1,192,2,2},x.options()).contiguous(torch::MemoryFormat::ChannelsLast);
            auto value=pixel_unshuffle_cuda(x,8,unshuffled);
            same(value,torch::pixel_unshuffle(x.cpu(),8),"FP16 unshuffle");
            auto restored=torch::empty_like(x);
            same(pixel_shuffle_8_cuda(value,false,restored),x.cpu(),"FP16 shuffle");
            same(pixel_shuffle_8_cuda(value,true,restored),x.cpu().clamp(-0.5,0.5),"FP16 shuffle clipping");
        });
        suite.run("FP16 broadcast multiply",[&]{
            auto device=torch::Device(torch::kCUDA,0);
            auto x=(torch::arange(256,torch::kFloat32)/128).reshape({1,16,4,4})
                .to(device,torch::kFloat16).contiguous(torch::MemoryFormat::ChannelsLast);
            auto scale=(torch::arange(16,torch::kFloat32)/16+0.5).to(device,torch::kFloat16);
            auto output=torch::empty_like(x);
            same(multiply_with_broadcast_cuda(x,scale,output),(x.cpu()*scale.cpu().reshape({1,16,1,1})),"FP16 multiply");
        });
        for(int stride:{1,2}) suite.run("FP16 spatial convolution stride="+std::to_string(stride),[&,stride]{
            auto device=torch::Device(torch::kCUDA,0);
            auto source_x=(torch::arange(16*6*8,torch::kFloat32).remainder(32)-16).reshape({1,16,6,8})/64;
            auto source_w=(torch::arange(16*16*9,torch::kFloat32).remainder(16)-8).reshape({16,16,3,3})/128;
            auto source_b=(torch::arange(16,torch::kFloat32)-8)/32;
            auto x=uf::fp16_parameter_to_device(source_x,device),w=uf::fp16_parameter_to_device(source_w,device);
            auto b=uf::fp16_parameter_to_device(source_b,device);
            auto expected=torch::conv2d(source_x,source_w,source_b,{stride,stride},{1,1}).to(torch::kFloat16);
            auto output=torch::empty({1,16,6/stride,8/stride},x.options()).contiguous(torch::MemoryFormat::ChannelsLast);
            auto actual=conv_bias(87,x,w,b,stride,output).cpu();
            require(actual.sizes()==expected.sizes() && torch::allclose(actual,expected,0.003,0.003),"FP16 spatial convolution");
        });
        suite.run("FP16 depthwise convolution",[&]{
            auto device=torch::Device(torch::kCUDA,0);
            auto source_x=(torch::arange(16*6*8,torch::kFloat32).remainder(32)-16).reshape({1,16,6,8})/64;
            auto source_w=(torch::arange(16*9,torch::kFloat32).remainder(16)-8).reshape({16,1,3,3})/128;
            auto x=uf::fp16_parameter_to_device(source_x,device);
            auto reordered=uf::fp16_parameter_to_device(source_w.permute({1,0,2,3}),device);
            auto output=torch::empty_like(x);
            auto expected=torch::conv2d(source_x,source_w,torch::Tensor{},{1,1},{1,1},{1,1},16).to(torch::kFloat16);
            require(torch::allclose(d3x3(87,x,reordered,output).cpu(),expected,0.003,0.003),"FP16 depthwise convolution");
        });
        suite.run("FP16 transposed convolution",[&]{
            auto device=torch::Device(torch::kCUDA,0);
            auto source_x=(torch::arange(16*3*4,torch::kFloat32).remainder(32)-16).reshape({1,16,3,4})/64;
            auto source_w=(torch::arange(16*16*4,torch::kFloat32).remainder(16)-8).reshape({16,16,2,2})/128;
            auto x=uf::fp16_parameter_to_device(source_x,device),w=uf::fp16_parameter_to_device(source_w,device);
            auto output=torch::empty({1,16,6,8},x.options()).contiguous(torch::MemoryFormat::ChannelsLast);
            auto expected=torch::conv_transpose2d(source_x,source_w,torch::Tensor{},{2,2}).to(torch::kFloat16);
            require(torch::allclose(transposed_conv(87,x,w,2,output).cpu(),expected,0.003,0.003),"FP16 transposed convolution");
        });
        for(bool bias:{false,true})suite.run("FP16 subpixel proxy synthetic parameter setup bias="+std::to_string(bias),[&,bias]{
            auto device=torch::Device(torch::kCUDA,0);
            auto source_x=(torch::arange(16*3*4,torch::kFloat32).remainder(32)-16).reshape({1,16,3,4})/64;
            auto source_w=(torch::arange(64*16,torch::kFloat32).remainder(16)-8).reshape({64,16,1,1})/128;
            auto source_b=(torch::arange(64,torch::kFloat32)-32)/128;
            auto x=uf::fp16_parameter_to_device(source_x,device);
            std::map<std::string,torch::Tensor> parameters{{"conv.0.weight",uf::fp16_parameter_to_device(source_w,device)}};
            if(bias)parameters.emplace("conv.0.bias",uf::fp16_parameter_to_device(source_b,device));
            SubpelConv2xProxy proxy;proxy.set_param(parameters);proxy.pre_allocate_tensors(x);
            auto expected=torch::pixel_shuffle(torch::conv2d(source_x,source_w,bias?source_b:torch::Tensor{}),2).to(torch::kFloat16);
            auto actual=proxy.forward(x).cpu();
            require(actual.sizes()==expected.sizes() && torch::allclose(actual,expected,0.003,0.003),"Subpixel transformed-parameter output");
            proxy.release_tensors();
        });
        suite.run("FP16 depth block folded bias and residual setup",[&]{
            auto device=torch::Device(torch::kCUDA,0);
            std::map<std::string,torch::Tensor> cpu,parameters;
            const auto add=[&](const std::string& name,const std::vector<std::int64_t>& shape){
                std::int64_t count=1;for(auto size:shape)count*=size;
                auto value=(torch::arange(count,torch::kFloat32).remainder(16)-8).reshape(shape)/128;
                cpu.emplace(name,value);parameters.emplace(name,uf::fp16_parameter_to_device(value,device));
            };
            for(const auto& name:{"dc.0","dc.3","ffn.2"}){add(std::string(name)+".weight",{16,16,1,1});add(std::string(name)+".bias",{16});}
            add("dc.2.weight",{16,1,3,3});add("dc.2.bias",{16});
            add("ffn.0.weight",{64,16,1,1});add("ffn.0.bias",{64});
            auto source_x=(torch::arange(16*4*4,torch::kFloat32).remainder(32)-16).reshape({1,16,4,4})/64;
            const auto convolution=[&](const torch::Tensor& x,const std::string& name,int groups=1){
                const int padding=cpu.at(name+".weight").size(2)/2;
                return torch::conv2d(x.to(torch::kFloat32),cpu.at(name+".weight"),cpu.at(name+".bias"),{1,1},{padding,padding},{1,1},groups).to(torch::kFloat16);
            };
            const auto activate=[](const torch::Tensor& x){return ((4*x.to(torch::kFloat32)).sigmoid()*x.to(torch::kFloat32)).to(torch::kFloat16);};
            auto expected=activate(convolution(source_x,"dc.0"));
            expected=convolution(expected,"dc.2",16);expected=(convolution(expected,"dc.3")+source_x).to(torch::kFloat16);
            auto ffn=activate(convolution(expected,"ffn.0"));
            ffn=ffn.reshape({1,16,4,4,4}).sum(2).to(torch::kFloat16);
            expected=(convolution(ffn,"ffn.2")+expected).to(torch::kFloat16);
            auto x=uf::fp16_parameter_to_device(source_x,device);
            DepthConvBlockProxy proxy;proxy.set_param(parameters,false);proxy.pre_allocate_tensors(x,true);
            auto actual=proxy.forward(x).cpu();require(torch::allclose(actual,expected,0.004,0.004),"Folded bias/residual synthetic block");
            proxy.release_tensors();
        });
    }
#endif
    if(cuda)torch::cuda::synchronize(0);
    std::filesystem::remove_all(temp);
    std::cout << suite.passed << "/" << suite.passed+suite.failed << " native checks passed; no models/checkpoints loaded" << std::endl;
    return suite.failed ? 1 : 0;
}
