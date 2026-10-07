// UF's configured one-bar progress display adapts tqdm 4.67.3 std.py/utils.py.
// Copyright (c) 2015-2026 Casper da Costa-Luis and tqdm contributors.
// SPDX-License-Identifier: MPL-2.0 AND MIT
// Retained notice: licenses/tqdm-LICENCE. No Python runtime is used.
#pragma once

#include "uf/python_json.hpp"
#include "uf/python_numeric.hpp"
#include "uf/process.hpp"
#include "uf/utf8.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <string_view>
#include <utility>
#include <sys/ioctl.h>
#include <unistd.h>

namespace uf::pareto_detail {

// Unicode 13 East Asian F/W intervals exported from reference CPython 3.10.20.
// Source data only; used for tqdm display-cell width, including unassigned ranges.
inline constexpr std::array<std::pair<std::uint32_t, std::uint32_t>, 692> progress_wide_ranges{{
    {0x000378u, 0x000379u},
    {0x000380u, 0x000383u},
    {0x00038bu, 0x00038bu},
    {0x00038du, 0x00038du},
    {0x0003a2u, 0x0003a2u},
    {0x000530u, 0x000530u},
    {0x000557u, 0x000558u},
    {0x00058bu, 0x00058cu},
    {0x000590u, 0x000590u},
    {0x0005c8u, 0x0005cfu},
    {0x0005ebu, 0x0005eeu},
    {0x0005f5u, 0x0005ffu},
    {0x00061du, 0x00061du},
    {0x00070eu, 0x00070eu},
    {0x00074bu, 0x00074cu},
    {0x0007b2u, 0x0007bfu},
    {0x0007fbu, 0x0007fcu},
    {0x00082eu, 0x00082fu},
    {0x00083fu, 0x00083fu},
    {0x00085cu, 0x00085du},
    {0x00085fu, 0x00085fu},
    {0x00086bu, 0x00089fu},
    {0x0008b5u, 0x0008b5u},
    {0x0008c8u, 0x0008d2u},
    {0x000984u, 0x000984u},
    {0x00098du, 0x00098eu},
    {0x000991u, 0x000992u},
    {0x0009a9u, 0x0009a9u},
    {0x0009b1u, 0x0009b1u},
    {0x0009b3u, 0x0009b5u},
    {0x0009bau, 0x0009bbu},
    {0x0009c5u, 0x0009c6u},
    {0x0009c9u, 0x0009cau},
    {0x0009cfu, 0x0009d6u},
    {0x0009d8u, 0x0009dbu},
    {0x0009deu, 0x0009deu},
    {0x0009e4u, 0x0009e5u},
    {0x0009ffu, 0x000a00u},
    {0x000a04u, 0x000a04u},
    {0x000a0bu, 0x000a0eu},
    {0x000a11u, 0x000a12u},
    {0x000a29u, 0x000a29u},
    {0x000a31u, 0x000a31u},
    {0x000a34u, 0x000a34u},
    {0x000a37u, 0x000a37u},
    {0x000a3au, 0x000a3bu},
    {0x000a3du, 0x000a3du},
    {0x000a43u, 0x000a46u},
    {0x000a49u, 0x000a4au},
    {0x000a4eu, 0x000a50u},
    {0x000a52u, 0x000a58u},
    {0x000a5du, 0x000a5du},
    {0x000a5fu, 0x000a65u},
    {0x000a77u, 0x000a80u},
    {0x000a84u, 0x000a84u},
    {0x000a8eu, 0x000a8eu},
    {0x000a92u, 0x000a92u},
    {0x000aa9u, 0x000aa9u},
    {0x000ab1u, 0x000ab1u},
    {0x000ab4u, 0x000ab4u},
    {0x000abau, 0x000abbu},
    {0x000ac6u, 0x000ac6u},
    {0x000acau, 0x000acau},
    {0x000aceu, 0x000acfu},
    {0x000ad1u, 0x000adfu},
    {0x000ae4u, 0x000ae5u},
    {0x000af2u, 0x000af8u},
    {0x000b00u, 0x000b00u},
    {0x000b04u, 0x000b04u},
    {0x000b0du, 0x000b0eu},
    {0x000b11u, 0x000b12u},
    {0x000b29u, 0x000b29u},
    {0x000b31u, 0x000b31u},
    {0x000b34u, 0x000b34u},
    {0x000b3au, 0x000b3bu},
    {0x000b45u, 0x000b46u},
    {0x000b49u, 0x000b4au},
    {0x000b4eu, 0x000b54u},
    {0x000b58u, 0x000b5bu},
    {0x000b5eu, 0x000b5eu},
    {0x000b64u, 0x000b65u},
    {0x000b78u, 0x000b81u},
    {0x000b84u, 0x000b84u},
    {0x000b8bu, 0x000b8du},
    {0x000b91u, 0x000b91u},
    {0x000b96u, 0x000b98u},
    {0x000b9bu, 0x000b9bu},
    {0x000b9du, 0x000b9du},
    {0x000ba0u, 0x000ba2u},
    {0x000ba5u, 0x000ba7u},
    {0x000babu, 0x000badu},
    {0x000bbau, 0x000bbdu},
    {0x000bc3u, 0x000bc5u},
    {0x000bc9u, 0x000bc9u},
    {0x000bceu, 0x000bcfu},
    {0x000bd1u, 0x000bd6u},
    {0x000bd8u, 0x000be5u},
    {0x000bfbu, 0x000bffu},
    {0x000c0du, 0x000c0du},
    {0x000c11u, 0x000c11u},
    {0x000c29u, 0x000c29u},
    {0x000c3au, 0x000c3cu},
    {0x000c45u, 0x000c45u},
    {0x000c49u, 0x000c49u},
    {0x000c4eu, 0x000c54u},
    {0x000c57u, 0x000c57u},
    {0x000c5bu, 0x000c5fu},
    {0x000c64u, 0x000c65u},
    {0x000c70u, 0x000c76u},
    {0x000c8du, 0x000c8du},
    {0x000c91u, 0x000c91u},
    {0x000ca9u, 0x000ca9u},
    {0x000cb4u, 0x000cb4u},
    {0x000cbau, 0x000cbbu},
    {0x000cc5u, 0x000cc5u},
    {0x000cc9u, 0x000cc9u},
    {0x000cceu, 0x000cd4u},
    {0x000cd7u, 0x000cddu},
    {0x000cdfu, 0x000cdfu},
    {0x000ce4u, 0x000ce5u},
    {0x000cf0u, 0x000cf0u},
    {0x000cf3u, 0x000cffu},
    {0x000d0du, 0x000d0du},
    {0x000d11u, 0x000d11u},
    {0x000d45u, 0x000d45u},
    {0x000d49u, 0x000d49u},
    {0x000d50u, 0x000d53u},
    {0x000d64u, 0x000d65u},
    {0x000d80u, 0x000d80u},
    {0x000d84u, 0x000d84u},
    {0x000d97u, 0x000d99u},
    {0x000db2u, 0x000db2u},
    {0x000dbcu, 0x000dbcu},
    {0x000dbeu, 0x000dbfu},
    {0x000dc7u, 0x000dc9u},
    {0x000dcbu, 0x000dceu},
    {0x000dd5u, 0x000dd5u},
    {0x000dd7u, 0x000dd7u},
    {0x000de0u, 0x000de5u},
    {0x000df0u, 0x000df1u},
    {0x000df5u, 0x000e00u},
    {0x000e3bu, 0x000e3eu},
    {0x000e5cu, 0x000e80u},
    {0x000e83u, 0x000e83u},
    {0x000e85u, 0x000e85u},
    {0x000e8bu, 0x000e8bu},
    {0x000ea4u, 0x000ea4u},
    {0x000ea6u, 0x000ea6u},
    {0x000ebeu, 0x000ebfu},
    {0x000ec5u, 0x000ec5u},
    {0x000ec7u, 0x000ec7u},
    {0x000eceu, 0x000ecfu},
    {0x000edau, 0x000edbu},
    {0x000ee0u, 0x000effu},
    {0x000f48u, 0x000f48u},
    {0x000f6du, 0x000f70u},
    {0x000f98u, 0x000f98u},
    {0x000fbdu, 0x000fbdu},
    {0x000fcdu, 0x000fcdu},
    {0x000fdbu, 0x000fffu},
    {0x0010c6u, 0x0010c6u},
    {0x0010c8u, 0x0010ccu},
    {0x0010ceu, 0x0010cfu},
    {0x001100u, 0x00115fu},
    {0x001249u, 0x001249u},
    {0x00124eu, 0x00124fu},
    {0x001257u, 0x001257u},
    {0x001259u, 0x001259u},
    {0x00125eu, 0x00125fu},
    {0x001289u, 0x001289u},
    {0x00128eu, 0x00128fu},
    {0x0012b1u, 0x0012b1u},
    {0x0012b6u, 0x0012b7u},
    {0x0012bfu, 0x0012bfu},
    {0x0012c1u, 0x0012c1u},
    {0x0012c6u, 0x0012c7u},
    {0x0012d7u, 0x0012d7u},
    {0x001311u, 0x001311u},
    {0x001316u, 0x001317u},
    {0x00135bu, 0x00135cu},
    {0x00137du, 0x00137fu},
    {0x00139au, 0x00139fu},
    {0x0013f6u, 0x0013f7u},
    {0x0013feu, 0x0013ffu},
    {0x00169du, 0x00169fu},
    {0x0016f9u, 0x0016ffu},
    {0x00170du, 0x00170du},
    {0x001715u, 0x00171fu},
    {0x001737u, 0x00173fu},
    {0x001754u, 0x00175fu},
    {0x00176du, 0x00176du},
    {0x001771u, 0x001771u},
    {0x001774u, 0x00177fu},
    {0x0017deu, 0x0017dfu},
    {0x0017eau, 0x0017efu},
    {0x0017fau, 0x0017ffu},
    {0x00180fu, 0x00180fu},
    {0x00181au, 0x00181fu},
    {0x001879u, 0x00187fu},
    {0x0018abu, 0x0018afu},
    {0x0018f6u, 0x0018ffu},
    {0x00191fu, 0x00191fu},
    {0x00192cu, 0x00192fu},
    {0x00193cu, 0x00193fu},
    {0x001941u, 0x001943u},
    {0x00196eu, 0x00196fu},
    {0x001975u, 0x00197fu},
    {0x0019acu, 0x0019afu},
    {0x0019cau, 0x0019cfu},
    {0x0019dbu, 0x0019ddu},
    {0x001a1cu, 0x001a1du},
    {0x001a5fu, 0x001a5fu},
    {0x001a7du, 0x001a7eu},
    {0x001a8au, 0x001a8fu},
    {0x001a9au, 0x001a9fu},
    {0x001aaeu, 0x001aafu},
    {0x001ac1u, 0x001affu},
    {0x001b4cu, 0x001b4fu},
    {0x001b7du, 0x001b7fu},
    {0x001bf4u, 0x001bfbu},
    {0x001c38u, 0x001c3au},
    {0x001c4au, 0x001c4cu},
    {0x001c89u, 0x001c8fu},
    {0x001cbbu, 0x001cbcu},
    {0x001cc8u, 0x001ccfu},
    {0x001cfbu, 0x001cffu},
    {0x001dfau, 0x001dfau},
    {0x001f16u, 0x001f17u},
    {0x001f1eu, 0x001f1fu},
    {0x001f46u, 0x001f47u},
    {0x001f4eu, 0x001f4fu},
    {0x001f58u, 0x001f58u},
    {0x001f5au, 0x001f5au},
    {0x001f5cu, 0x001f5cu},
    {0x001f5eu, 0x001f5eu},
    {0x001f7eu, 0x001f7fu},
    {0x001fb5u, 0x001fb5u},
    {0x001fc5u, 0x001fc5u},
    {0x001fd4u, 0x001fd5u},
    {0x001fdcu, 0x001fdcu},
    {0x001ff0u, 0x001ff1u},
    {0x001ff5u, 0x001ff5u},
    {0x001fffu, 0x001fffu},
    {0x002065u, 0x002065u},
    {0x002072u, 0x002073u},
    {0x00208fu, 0x00208fu},
    {0x00209du, 0x00209fu},
    {0x0020c0u, 0x0020cfu},
    {0x0020f1u, 0x0020ffu},
    {0x00218cu, 0x00218fu},
    {0x00231au, 0x00231bu},
    {0x002329u, 0x00232au},
    {0x0023e9u, 0x0023ecu},
    {0x0023f0u, 0x0023f0u},
    {0x0023f3u, 0x0023f3u},
    {0x002427u, 0x00243fu},
    {0x00244bu, 0x00245fu},
    {0x0025fdu, 0x0025feu},
    {0x002614u, 0x002615u},
    {0x002648u, 0x002653u},
    {0x00267fu, 0x00267fu},
    {0x002693u, 0x002693u},
    {0x0026a1u, 0x0026a1u},
    {0x0026aau, 0x0026abu},
    {0x0026bdu, 0x0026beu},
    {0x0026c4u, 0x0026c5u},
    {0x0026ceu, 0x0026ceu},
    {0x0026d4u, 0x0026d4u},
    {0x0026eau, 0x0026eau},
    {0x0026f2u, 0x0026f3u},
    {0x0026f5u, 0x0026f5u},
    {0x0026fau, 0x0026fau},
    {0x0026fdu, 0x0026fdu},
    {0x002705u, 0x002705u},
    {0x00270au, 0x00270bu},
    {0x002728u, 0x002728u},
    {0x00274cu, 0x00274cu},
    {0x00274eu, 0x00274eu},
    {0x002753u, 0x002755u},
    {0x002757u, 0x002757u},
    {0x002795u, 0x002797u},
    {0x0027b0u, 0x0027b0u},
    {0x0027bfu, 0x0027bfu},
    {0x002b1bu, 0x002b1cu},
    {0x002b50u, 0x002b50u},
    {0x002b55u, 0x002b55u},
    {0x002b74u, 0x002b75u},
    {0x002b96u, 0x002b96u},
    {0x002c2fu, 0x002c2fu},
    {0x002c5fu, 0x002c5fu},
    {0x002cf4u, 0x002cf8u},
    {0x002d26u, 0x002d26u},
    {0x002d28u, 0x002d2cu},
    {0x002d2eu, 0x002d2fu},
    {0x002d68u, 0x002d6eu},
    {0x002d71u, 0x002d7eu},
    {0x002d97u, 0x002d9fu},
    {0x002da7u, 0x002da7u},
    {0x002dafu, 0x002dafu},
    {0x002db7u, 0x002db7u},
    {0x002dbfu, 0x002dbfu},
    {0x002dc7u, 0x002dc7u},
    {0x002dcfu, 0x002dcfu},
    {0x002dd7u, 0x002dd7u},
    {0x002ddfu, 0x002ddfu},
    {0x002e53u, 0x00303eu},
    {0x003040u, 0x003247u},
    {0x003250u, 0x004dbfu},
    {0x004e00u, 0x00a4cfu},
    {0x00a62cu, 0x00a63fu},
    {0x00a6f8u, 0x00a6ffu},
    {0x00a7c0u, 0x00a7c1u},
    {0x00a7cbu, 0x00a7f4u},
    {0x00a82du, 0x00a82fu},
    {0x00a83au, 0x00a83fu},
    {0x00a878u, 0x00a87fu},
    {0x00a8c6u, 0x00a8cdu},
    {0x00a8dau, 0x00a8dfu},
    {0x00a954u, 0x00a95eu},
    {0x00a960u, 0x00a97fu},
    {0x00a9ceu, 0x00a9ceu},
    {0x00a9dau, 0x00a9ddu},
    {0x00a9ffu, 0x00a9ffu},
    {0x00aa37u, 0x00aa3fu},
    {0x00aa4eu, 0x00aa4fu},
    {0x00aa5au, 0x00aa5bu},
    {0x00aac3u, 0x00aadau},
    {0x00aaf7u, 0x00ab00u},
    {0x00ab07u, 0x00ab08u},
    {0x00ab0fu, 0x00ab10u},
    {0x00ab17u, 0x00ab1fu},
    {0x00ab27u, 0x00ab27u},
    {0x00ab2fu, 0x00ab2fu},
    {0x00ab6cu, 0x00ab6fu},
    {0x00abeeu, 0x00abefu},
    {0x00abfau, 0x00d7afu},
    {0x00d7c7u, 0x00d7cau},
    {0x00d7fcu, 0x00d7ffu},
    {0x00f900u, 0x00faffu},
    {0x00fb07u, 0x00fb12u},
    {0x00fb18u, 0x00fb1cu},
    {0x00fb37u, 0x00fb37u},
    {0x00fb3du, 0x00fb3du},
    {0x00fb3fu, 0x00fb3fu},
    {0x00fb42u, 0x00fb42u},
    {0x00fb45u, 0x00fb45u},
    {0x00fbc2u, 0x00fbd2u},
    {0x00fd40u, 0x00fd4fu},
    {0x00fd90u, 0x00fd91u},
    {0x00fdc8u, 0x00fdefu},
    {0x00fdfeu, 0x00fdffu},
    {0x00fe10u, 0x00fe1fu},
    {0x00fe30u, 0x00fe6fu},
    {0x00fe75u, 0x00fe75u},
    {0x00fefdu, 0x00fefeu},
    {0x00ff00u, 0x00ff60u},
    {0x00ffbfu, 0x00ffc1u},
    {0x00ffc8u, 0x00ffc9u},
    {0x00ffd0u, 0x00ffd1u},
    {0x00ffd8u, 0x00ffd9u},
    {0x00ffddu, 0x00ffe7u},
    {0x00ffefu, 0x00fff8u},
    {0x00fffeu, 0x00ffffu},
    {0x01000cu, 0x01000cu},
    {0x010027u, 0x010027u},
    {0x01003bu, 0x01003bu},
    {0x01003eu, 0x01003eu},
    {0x01004eu, 0x01004fu},
    {0x01005eu, 0x01007fu},
    {0x0100fbu, 0x0100ffu},
    {0x010103u, 0x010106u},
    {0x010134u, 0x010136u},
    {0x01018fu, 0x01018fu},
    {0x01019du, 0x01019fu},
    {0x0101a1u, 0x0101cfu},
    {0x0101feu, 0x01027fu},
    {0x01029du, 0x01029fu},
    {0x0102d1u, 0x0102dfu},
    {0x0102fcu, 0x0102ffu},
    {0x010324u, 0x01032cu},
    {0x01034bu, 0x01034fu},
    {0x01037bu, 0x01037fu},
    {0x01039eu, 0x01039eu},
    {0x0103c4u, 0x0103c7u},
    {0x0103d6u, 0x0103ffu},
    {0x01049eu, 0x01049fu},
    {0x0104aau, 0x0104afu},
    {0x0104d4u, 0x0104d7u},
    {0x0104fcu, 0x0104ffu},
    {0x010528u, 0x01052fu},
    {0x010564u, 0x01056eu},
    {0x010570u, 0x0105ffu},
    {0x010737u, 0x01073fu},
    {0x010756u, 0x01075fu},
    {0x010768u, 0x0107ffu},
    {0x010806u, 0x010807u},
    {0x010809u, 0x010809u},
    {0x010836u, 0x010836u},
    {0x010839u, 0x01083bu},
    {0x01083du, 0x01083eu},
    {0x010856u, 0x010856u},
    {0x01089fu, 0x0108a6u},
    {0x0108b0u, 0x0108dfu},
    {0x0108f3u, 0x0108f3u},
    {0x0108f6u, 0x0108fau},
    {0x01091cu, 0x01091eu},
    {0x01093au, 0x01093eu},
    {0x010940u, 0x01097fu},
    {0x0109b8u, 0x0109bbu},
    {0x0109d0u, 0x0109d1u},
    {0x010a04u, 0x010a04u},
    {0x010a07u, 0x010a0bu},
    {0x010a14u, 0x010a14u},
    {0x010a18u, 0x010a18u},
    {0x010a36u, 0x010a37u},
    {0x010a3bu, 0x010a3eu},
    {0x010a49u, 0x010a4fu},
    {0x010a59u, 0x010a5fu},
    {0x010aa0u, 0x010abfu},
    {0x010ae7u, 0x010aeau},
    {0x010af7u, 0x010affu},
    {0x010b36u, 0x010b38u},
    {0x010b56u, 0x010b57u},
    {0x010b73u, 0x010b77u},
    {0x010b92u, 0x010b98u},
    {0x010b9du, 0x010ba8u},
    {0x010bb0u, 0x010bffu},
    {0x010c49u, 0x010c7fu},
    {0x010cb3u, 0x010cbfu},
    {0x010cf3u, 0x010cf9u},
    {0x010d28u, 0x010d2fu},
    {0x010d3au, 0x010e5fu},
    {0x010e7fu, 0x010e7fu},
    {0x010eaau, 0x010eaau},
    {0x010eaeu, 0x010eafu},
    {0x010eb2u, 0x010effu},
    {0x010f28u, 0x010f2fu},
    {0x010f5au, 0x010fafu},
    {0x010fccu, 0x010fdfu},
    {0x010ff7u, 0x010fffu},
    {0x01104eu, 0x011051u},
    {0x011070u, 0x01107eu},
    {0x0110c2u, 0x0110ccu},
    {0x0110ceu, 0x0110cfu},
    {0x0110e9u, 0x0110efu},
    {0x0110fau, 0x0110ffu},
    {0x011135u, 0x011135u},
    {0x011148u, 0x01114fu},
    {0x011177u, 0x01117fu},
    {0x0111e0u, 0x0111e0u},
    {0x0111f5u, 0x0111ffu},
    {0x011212u, 0x011212u},
    {0x01123fu, 0x01127fu},
    {0x011287u, 0x011287u},
    {0x011289u, 0x011289u},
    {0x01128eu, 0x01128eu},
    {0x01129eu, 0x01129eu},
    {0x0112aau, 0x0112afu},
    {0x0112ebu, 0x0112efu},
    {0x0112fau, 0x0112ffu},
    {0x011304u, 0x011304u},
    {0x01130du, 0x01130eu},
    {0x011311u, 0x011312u},
    {0x011329u, 0x011329u},
    {0x011331u, 0x011331u},
    {0x011334u, 0x011334u},
    {0x01133au, 0x01133au},
    {0x011345u, 0x011346u},
    {0x011349u, 0x01134au},
    {0x01134eu, 0x01134fu},
    {0x011351u, 0x011356u},
    {0x011358u, 0x01135cu},
    {0x011364u, 0x011365u},
    {0x01136du, 0x01136fu},
    {0x011375u, 0x0113ffu},
    {0x01145cu, 0x01145cu},
    {0x011462u, 0x01147fu},
    {0x0114c8u, 0x0114cfu},
    {0x0114dau, 0x01157fu},
    {0x0115b6u, 0x0115b7u},
    {0x0115deu, 0x0115ffu},
    {0x011645u, 0x01164fu},
    {0x01165au, 0x01165fu},
    {0x01166du, 0x01167fu},
    {0x0116b9u, 0x0116bfu},
    {0x0116cau, 0x0116ffu},
    {0x01171bu, 0x01171cu},
    {0x01172cu, 0x01172fu},
    {0x011740u, 0x0117ffu},
    {0x01183cu, 0x01189fu},
    {0x0118f3u, 0x0118feu},
    {0x011907u, 0x011908u},
    {0x01190au, 0x01190bu},
    {0x011914u, 0x011914u},
    {0x011917u, 0x011917u},
    {0x011936u, 0x011936u},
    {0x011939u, 0x01193au},
    {0x011947u, 0x01194fu},
    {0x01195au, 0x01199fu},
    {0x0119a8u, 0x0119a9u},
    {0x0119d8u, 0x0119d9u},
    {0x0119e5u, 0x0119ffu},
    {0x011a48u, 0x011a4fu},
    {0x011aa3u, 0x011abfu},
    {0x011af9u, 0x011bffu},
    {0x011c09u, 0x011c09u},
    {0x011c37u, 0x011c37u},
    {0x011c46u, 0x011c4fu},
    {0x011c6du, 0x011c6fu},
    {0x011c90u, 0x011c91u},
    {0x011ca8u, 0x011ca8u},
    {0x011cb7u, 0x011cffu},
    {0x011d07u, 0x011d07u},
    {0x011d0au, 0x011d0au},
    {0x011d37u, 0x011d39u},
    {0x011d3bu, 0x011d3bu},
    {0x011d3eu, 0x011d3eu},
    {0x011d48u, 0x011d4fu},
    {0x011d5au, 0x011d5fu},
    {0x011d66u, 0x011d66u},
    {0x011d69u, 0x011d69u},
    {0x011d8fu, 0x011d8fu},
    {0x011d92u, 0x011d92u},
    {0x011d99u, 0x011d9fu},
    {0x011daau, 0x011edfu},
    {0x011ef9u, 0x011fafu},
    {0x011fb1u, 0x011fbfu},
    {0x011ff2u, 0x011ffeu},
    {0x01239au, 0x0123ffu},
    {0x01246fu, 0x01246fu},
    {0x012475u, 0x01247fu},
    {0x012544u, 0x012fffu},
    {0x01342fu, 0x01342fu},
    {0x013439u, 0x0143ffu},
    {0x014647u, 0x0167ffu},
    {0x016a39u, 0x016a3fu},
    {0x016a5fu, 0x016a5fu},
    {0x016a6au, 0x016a6du},
    {0x016a70u, 0x016acfu},
    {0x016aeeu, 0x016aefu},
    {0x016af6u, 0x016affu},
    {0x016b46u, 0x016b4fu},
    {0x016b5au, 0x016b5au},
    {0x016b62u, 0x016b62u},
    {0x016b78u, 0x016b7cu},
    {0x016b90u, 0x016e3fu},
    {0x016e9bu, 0x016effu},
    {0x016f4bu, 0x016f4eu},
    {0x016f88u, 0x016f8eu},
    {0x016fa0u, 0x01bbffu},
    {0x01bc6bu, 0x01bc6fu},
    {0x01bc7du, 0x01bc7fu},
    {0x01bc89u, 0x01bc8fu},
    {0x01bc9au, 0x01bc9bu},
    {0x01bca4u, 0x01cfffu},
    {0x01d0f6u, 0x01d0ffu},
    {0x01d127u, 0x01d128u},
    {0x01d1e9u, 0x01d1ffu},
    {0x01d246u, 0x01d2dfu},
    {0x01d2f4u, 0x01d2ffu},
    {0x01d357u, 0x01d35fu},
    {0x01d379u, 0x01d3ffu},
    {0x01d455u, 0x01d455u},
    {0x01d49du, 0x01d49du},
    {0x01d4a0u, 0x01d4a1u},
    {0x01d4a3u, 0x01d4a4u},
    {0x01d4a7u, 0x01d4a8u},
    {0x01d4adu, 0x01d4adu},
    {0x01d4bau, 0x01d4bau},
    {0x01d4bcu, 0x01d4bcu},
    {0x01d4c4u, 0x01d4c4u},
    {0x01d506u, 0x01d506u},
    {0x01d50bu, 0x01d50cu},
    {0x01d515u, 0x01d515u},
    {0x01d51du, 0x01d51du},
    {0x01d53au, 0x01d53au},
    {0x01d53fu, 0x01d53fu},
    {0x01d545u, 0x01d545u},
    {0x01d547u, 0x01d549u},
    {0x01d551u, 0x01d551u},
    {0x01d6a6u, 0x01d6a7u},
    {0x01d7ccu, 0x01d7cdu},
    {0x01da8cu, 0x01da9au},
    {0x01daa0u, 0x01daa0u},
    {0x01dab0u, 0x01dfffu},
    {0x01e007u, 0x01e007u},
    {0x01e019u, 0x01e01au},
    {0x01e022u, 0x01e022u},
    {0x01e025u, 0x01e025u},
    {0x01e02bu, 0x01e0ffu},
    {0x01e12du, 0x01e12fu},
    {0x01e13eu, 0x01e13fu},
    {0x01e14au, 0x01e14du},
    {0x01e150u, 0x01e2bfu},
    {0x01e2fau, 0x01e2feu},
    {0x01e300u, 0x01e7ffu},
    {0x01e8c5u, 0x01e8c6u},
    {0x01e8d7u, 0x01e8ffu},
    {0x01e94cu, 0x01e94fu},
    {0x01e95au, 0x01e95du},
    {0x01e960u, 0x01ec70u},
    {0x01ecb5u, 0x01ed00u},
    {0x01ed3eu, 0x01edffu},
    {0x01ee04u, 0x01ee04u},
    {0x01ee20u, 0x01ee20u},
    {0x01ee23u, 0x01ee23u},
    {0x01ee25u, 0x01ee26u},
    {0x01ee28u, 0x01ee28u},
    {0x01ee33u, 0x01ee33u},
    {0x01ee38u, 0x01ee38u},
    {0x01ee3au, 0x01ee3au},
    {0x01ee3cu, 0x01ee41u},
    {0x01ee43u, 0x01ee46u},
    {0x01ee48u, 0x01ee48u},
    {0x01ee4au, 0x01ee4au},
    {0x01ee4cu, 0x01ee4cu},
    {0x01ee50u, 0x01ee50u},
    {0x01ee53u, 0x01ee53u},
    {0x01ee55u, 0x01ee56u},
    {0x01ee58u, 0x01ee58u},
    {0x01ee5au, 0x01ee5au},
    {0x01ee5cu, 0x01ee5cu},
    {0x01ee5eu, 0x01ee5eu},
    {0x01ee60u, 0x01ee60u},
    {0x01ee63u, 0x01ee63u},
    {0x01ee65u, 0x01ee66u},
    {0x01ee6bu, 0x01ee6bu},
    {0x01ee73u, 0x01ee73u},
    {0x01ee78u, 0x01ee78u},
    {0x01ee7du, 0x01ee7du},
    {0x01ee7fu, 0x01ee7fu},
    {0x01ee8au, 0x01ee8au},
    {0x01ee9cu, 0x01eea0u},
    {0x01eea4u, 0x01eea4u},
    {0x01eeaau, 0x01eeaau},
    {0x01eebcu, 0x01eeefu},
    {0x01eef2u, 0x01efffu},
    {0x01f004u, 0x01f004u},
    {0x01f02cu, 0x01f02fu},
    {0x01f094u, 0x01f09fu},
    {0x01f0afu, 0x01f0b0u},
    {0x01f0c0u, 0x01f0c0u},
    {0x01f0cfu, 0x01f0d0u},
    {0x01f0f6u, 0x01f0ffu},
    {0x01f18eu, 0x01f18eu},
    {0x01f191u, 0x01f19au},
    {0x01f1aeu, 0x01f1e5u},
    {0x01f200u, 0x01f320u},
    {0x01f32du, 0x01f335u},
    {0x01f337u, 0x01f37cu},
    {0x01f37eu, 0x01f393u},
    {0x01f3a0u, 0x01f3cau},
    {0x01f3cfu, 0x01f3d3u},
    {0x01f3e0u, 0x01f3f0u},
    {0x01f3f4u, 0x01f3f4u},
    {0x01f3f8u, 0x01f43eu},
    {0x01f440u, 0x01f440u},
    {0x01f442u, 0x01f4fcu},
    {0x01f4ffu, 0x01f53du},
    {0x01f54bu, 0x01f54eu},
    {0x01f550u, 0x01f567u},
    {0x01f57au, 0x01f57au},
    {0x01f595u, 0x01f596u},
    {0x01f5a4u, 0x01f5a4u},
    {0x01f5fbu, 0x01f64fu},
    {0x01f680u, 0x01f6c5u},
    {0x01f6ccu, 0x01f6ccu},
    {0x01f6d0u, 0x01f6d2u},
    {0x01f6d5u, 0x01f6dfu},
    {0x01f6ebu, 0x01f6efu},
    {0x01f6f4u, 0x01f6ffu},
    {0x01f774u, 0x01f77fu},
    {0x01f7d9u, 0x01f7ffu},
    {0x01f80cu, 0x01f80fu},
    {0x01f848u, 0x01f84fu},
    {0x01f85au, 0x01f85fu},
    {0x01f888u, 0x01f88fu},
    {0x01f8aeu, 0x01f8afu},
    {0x01f8b2u, 0x01f8ffu},
    {0x01f90cu, 0x01f93au},
    {0x01f93cu, 0x01f945u},
    {0x01f947u, 0x01f9ffu},
    {0x01fa54u, 0x01fa5fu},
    {0x01fa6eu, 0x01faffu},
    {0x01fb93u, 0x01fb93u},
    {0x01fbcbu, 0x01fbefu},
    {0x01fbfau, 0x0e0000u},
    {0x0e0002u, 0x0e001fu},
    {0x0e0080u, 0x0e00ffu},
    {0x0e01f0u, 0x0effffu},
    {0x0ffffeu, 0x0fffffu},
    {0x10fffeu, 0x10ffffu},
}};


// The UF caller supplies total/initial/unit/description and dynamic columns.
// Keep progress on stderr and state/result publication in the caller.
class ParetoProgress {
public:
    ParetoProgress(std::uint64_t total, std::uint64_t initial)
        : total_(total), initial_(initial), count_(initial), last_count_(initial) {
        if (!total || initial > total) throw std::invalid_argument("Invalid Pareto progress counts");
        mininterval_ = environment_float("TQDM_MININTERVAL", 0.1);
        maxinterval_ = environment_float("TQDM_MAXINTERVAL", 10.0);
        smoothing_ = environment_float("TQDM_SMOOTHING", 0.3);
        delay_ = environment_float("TQDM_DELAY", 0.0);
        if (const auto* miniters = std::getenv("TQDM_MINITERS")) {
            miniters_ = python_float(miniters);
            dynamic_miniters_ = false;
        }
        if (const auto* leave = std::getenv("TQDM_LEAVE")) leave_ = *leave;
        const auto* disabled = std::getenv("TQDM_DISABLE");
        disabled_ = disabled && *disabled;  // bool(str), as envwrap does.
        if (!disabled_) {
            // tqdm's status-printer setup flushes the original standard streams
            // before its wrapped writes, even when the first display is delayed.
            std::cerr.flush();
            if (!std::cerr) throw std::runtime_error("Cannot flush Pareto progress stderr");
            std::cout.flush();
            if (!std::cout) throw std::runtime_error("Cannot flush Pareto progress stdout");
        }
        if (!disabled_ && delay_ <= 0) refresh(false);
        // tqdm starts its clock after the initial display, whose elapsed is zero.
        started_ = last_time_ = now();
        initialized_ = true;
        if (!disabled_) {
            try { monitor_ = std::thread([this] { monitor_loop(); }); }
            catch (const std::exception& error) {
                // Like tqdm, failure to start its optional monitor leaves the
                // foreground progress display available.
                std::cerr << "Pareto progress: disabling monitor support: " << error.what() << '\n';
                if (!std::cerr) throw std::runtime_error("Cannot report Pareto monitor failure");
            }
        }
    }

    ~ParetoProgress() { try { close(); } catch (...) {} }

    void update(const nlohmann::json& result, std::size_t frontier) {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        if (disabled_ || closed_) return;
        ++count_;
        const auto completed = static_cast<double>(count_ - last_count_);
        if (completed >= miniters_) {
            const auto sampled = now();
            const auto delta = sampled - last_time_;
            if (delta >= mininterval_ && sampled >= started_ + delay_) {
                const auto sampled_again = now();
                if (smoothing_ != 0 && delta != 0 && completed != 0) {
                    ema_count_ = smoothing_ * completed + (1.0 - smoothing_) * ema_count_;
                    ema_time_ = smoothing_ * delta + (1.0 - smoothing_) * ema_time_;
                    ++ema_samples_;
                }
                refresh(false);  // update() refreshes with the previous postfix.
                if (dynamic_miniters_) {
                    if (maxinterval_ != 0 && delta >= maxinterval_) {
                        if (delta == 0) throw std::runtime_error("Zero Pareto progress interval divisor");
                        miniters_ = completed * (mininterval_ != 0 ? mininterval_ : maxinterval_) / delta;
                    } else if (smoothing_ != 0) {
                        const auto target = completed * (mininterval_ != 0 && delta != 0
                            ? mininterval_ / delta : 1.0);
                        ema_miniters_ = smoothing_ * target + (1.0 - smoothing_) * ema_miniters_;
                        ++miniter_samples_;
                        miniters_ = ema_value(ema_miniters_, miniter_samples_);
                    } else miniters_ = std::max(miniters_, completed);
                }
                last_count_ = count_;
                last_time_ = sampled_again;
            }
        }
        if (result.at("status") == "ok") {
            std::ostringstream quality;
            quality.imbue(std::locale::classic());
            quality << std::fixed << std::setprecision(3)
                    << python_json_as_double(result.at("psnr_db"));
            postfix_ = "bytes=" + concise(result.at("bitstream_bytes").get<std::uint64_t>()) +
                ", frontier=" + concise(frontier) + ", psnr=" + quality.str();
        } else {
            const auto error = utf8_replace(result.at("error").get_ref<const std::string&>());
            std::size_t end = 0, characters = 0;
            while (end < error.size() && characters < 40) {
                ++end;
                while (end < error.size() &&
                       (static_cast<unsigned char>(error[end]) & 0xc0) == 0x80) ++end;
                ++characters;
            }
            postfix_ = "error=" + python_strip(std::string_view(error.data(), end));
        }
        // set_postfix(refresh=True) displays every completed UF trial.
        refresh(false);
    }

    void close() {
        stop_monitor();
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        if (closed_) return;
        closed_ = true;
        if (disabled_) return;
        if (last_time_ < started_ + delay_) return;
        if (leave_) {
            refresh(true);  // Final rate uses the overall completed/elapsed ratio.
            write_display("\n");
        } else {
            write_display("\r" + std::string(last_width_, ' '));
            write_display("\r");
            last_width_ = 0;
        }
    }

private:
    std::uint64_t total_, initial_, count_, last_count_;
    double started_{}, last_time_{}, ema_count_{}, ema_time_{}, ema_miniters_{}, miniters_{};
    double mininterval_{0.1}, maxinterval_{10.0}, smoothing_{0.3}, delay_{};
    std::uint64_t miniter_samples_{}, ema_samples_{};
    std::size_t last_width_{};
    bool disabled_{}, closed_{}, initialized_{}, dynamic_miniters_{true}, leave_{true};
    std::string postfix_;
    std::recursive_mutex mutex_;
    std::condition_variable_any monitor_event_;
    bool monitor_stop_{};
    std::thread monitor_;
    const std::thread::id owner_thread_{std::this_thread::get_id()};

    void write_display(std::string_view text) {
        std::size_t offset = 0;
        while (offset < text.size()) {
            const auto remaining = text.size() - offset;
            const auto amount = std::min(remaining,
                static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
            const auto written = ::write(STDERR_FILENO, text.data() + offset, amount);
            if (written > 0) { offset += static_cast<std::size_t>(written); continue; }
            if (written == 0) throw std::runtime_error("Zero-length Pareto progress write");
            const auto error = errno;
            if (error == EINTR) {
                if (std::this_thread::get_id() == owner_thread_) ProcessInterrupt::check();
                continue;
            }
            if (error == EIO) {
                // tqdm's DisableOnWriteError suppresses EIO and raises the
                // iteration threshold. Explicit postfix/close writes still run.
                miniters_ = std::numeric_limits<double>::infinity();
                return;
            }
            throw std::runtime_error("Cannot write Pareto progress display (errno " +
                                     std::to_string(error) + ")");
        }
    }

    void stop_monitor() {
        {
            std::lock_guard<std::recursive_mutex> lock(mutex_);
            monitor_stop_ = true;
            monitor_event_.notify_all();
        }
        // The worker needs mutex_ to observe stop, so join outside the lock.
        if (monitor_.joinable()) monitor_.join();
    }

    void monitor_loop() noexcept {
        try {
            std::unique_lock<std::recursive_mutex> lock(mutex_);
            while (!monitor_event_.wait_for(lock, std::chrono::seconds(10),
                                            [this] { return monitor_stop_; })) {
                if (!disabled_ && !closed_ && miniters_ > 1 &&
                    now() - last_time_ >= maxinterval_) {
                    miniters_ = 1;
                    refresh(false);
                }
            }
        } catch (...) {
            // An exception terminates tqdm's optional monitor thread as well;
            // it must not escape a C++ thread entry and terminate the process.
            // Foreground update/close retain their checked error propagation.
        }
    }

    static double environment_float(const char* name, double fallback) {
        const auto* value = std::getenv(name);
        return value ? python_float(value) : fallback;
    }

    double ema_value(double value, std::uint64_t samples) const {
        if (!samples) return value;
        const auto correction = 1.0 - std::pow(1.0 - smoothing_, samples);
        if (correction == 0) throw std::runtime_error("Zero Pareto progress EMA divisor");
        return value / correction;
    }

    static double now() {
        return std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }

    static std::string concise(std::uint64_t value) {
        const auto integer = std::to_string(value);
        std::array<char, 64> text{};
        const auto converted = std::to_chars(text.data(), text.data() + text.size(),
            static_cast<double>(value), std::chars_format::general, 3);
        if (converted.ec != std::errc{}) throw std::runtime_error("Cannot format Pareto progress number");
        std::string compact(text.data(), converted.ptr);
        for (const auto* exponent : {"e+0", "e-0"}) {
            const auto at = compact.find(exponent);
            if (at != std::string::npos) compact.erase(at + 2, 1);
        }
        return compact.size() < integer.size() ? compact : integer;
    }

    static std::string interval(double seconds) {
        if (!std::isfinite(seconds) || std::abs(seconds) >= std::ldexp(1.0, 63))
            throw std::overflow_error("Pareto progress interval exceeds native range");
        const auto whole = static_cast<std::int64_t>(seconds);
        const auto magnitude = static_cast<std::uint64_t>(whole < 0 ? -whole : whole);
        std::ostringstream output;
        output.imbue(std::locale::classic());
        if (seconds < 0) output << '-';
        if (magnitude / 3600) output << magnitude / 3600 << ':';
        output << std::setfill('0') << std::setw(2) << (magnitude / 60) % 60
               << ':' << std::setw(2) << magnitude % 60;
        return output.str();
    }

    static std::size_t ansi_end(const std::string& text, std::size_t offset) {
        if (text[offset] != '\x1b' || offset + 1 >= text.size() || text[offset + 1] != '[')
            return offset;
        auto end = offset + 2;
        while (end < text.size() && (text[end] == ';' || (text[end] >= '0' && text[end] <= '9'))) ++end;
        if (end < text.size() && ((text[end] >= 'A' && text[end] <= 'Z') ||
                                 (text[end] >= 'a' && text[end] <= 'z'))) return end + 1;
        return offset;
    }

    static std::uint32_t codepoint(const std::string& text, std::size_t& offset) {
        const auto first = static_cast<unsigned char>(text[offset++]);
        if (first < 0x80) return first;
        const auto extra = first < 0xe0 ? 1u : first < 0xf0 ? 2u : 3u;
        std::uint32_t value = first & (extra == 1 ? 0x1fu : extra == 2 ? 0x0fu : 0x07u);
        for (unsigned i = 0; i < extra; ++i) {
            if (offset >= text.size()) throw std::invalid_argument("Incomplete progress UTF-8 text");
            const auto byte = static_cast<unsigned char>(text[offset++]);
            if ((byte & 0xc0) != 0x80) throw std::invalid_argument("Invalid progress UTF-8 text");
            value = (value << 6u) | (byte & 0x3fu);
        }
        return value;
    }

    static std::size_t cell_width(std::uint32_t value) {
        const auto found = std::lower_bound(progress_wide_ranges.begin(), progress_wide_ranges.end(), value,
            [](const auto& range, auto point) { return range.second < point; });
        return found != progress_wide_ranges.end() && found->first <= value ? 2 : 1;
    }

    static std::size_t display_width(const std::string& text) {
        std::size_t width = 0;
        for (std::size_t offset = 0; offset < text.size();) {
            const auto escape = ansi_end(text, offset);
            if (escape != offset) { offset = escape; continue; }
            width += cell_width(codepoint(text, offset));
        }
        return width;
    }

    static std::string trim(const std::string& text, std::size_t limit) {
        std::size_t offset = 0, kept = 0, width = 0;
        bool ansi = false;
        while (offset < text.size()) {
            const auto escape = ansi_end(text, offset);
            if (escape != offset) { ansi = true; offset = escape; kept = offset; continue; }
            const auto next = cell_width(codepoint(text, offset));
            if (next > limit - width) break;
            width += next; kept = offset;
        }
        auto result = text.substr(0, kept);
        if (ansi && result.find('\x1b') != std::string::npos &&
            (result.size() < 4 || result.compare(result.size() - 4, 4, "\x1b[0m"))) result += "\x1b[0m";
        return result;
    }

    static std::optional<std::size_t> terminal_columns() {
        winsize size{};
        if (ioctl(STDERR_FILENO, TIOCGWINSZ, &size) == 0) return size.ws_col;
        const auto* columns = std::getenv("COLUMNS");
        const auto* lines = std::getenv("LINES");
        if (!columns || !lines) return std::nullopt;
        try {
            (void)python_json_int(nlohmann::json(std::string(lines)));
            const auto count = python_json_index_int64(python_json_integer(
                python_json_int(nlohmann::json(std::string(columns)))));
            // Negative display widths have no useful native terminal meaning.
            // Keep zero distinct from unknown: tqdm's ncols=0 omits the bar.
            return count > 0 ? std::optional<std::size_t>(static_cast<std::size_t>(count - 1))
                             : std::nullopt;
        } catch (const std::invalid_argument&) { return std::nullopt; }
        catch (const std::overflow_error&) { return std::nullopt; }
        catch (const std::runtime_error&) { return std::nullopt; }
    }

    void refresh(bool final) {
        const auto elapsed = initialized_ ? now() - started_ : 0.0;
        const auto average_time = final ? 0.0 : ema_value(ema_time_, ema_samples_);
        const auto average_count = !final && average_time != 0
            ? ema_value(ema_count_, ema_samples_) : 0.0;
        const auto rate = !final && average_time != 0 ? average_count / average_time
            : elapsed != 0 ? static_cast<double>(count_ - initial_) / elapsed : 0.0;
        std::ostringstream speed;
        speed.imbue(std::locale::classic());
        if (rate == 0) speed << "?config/s";
        else if (1.0 / rate > 1) speed << std::fixed << std::setw(5)
            << std::setprecision(2) << 1.0 / rate << "s/config";
        else speed << std::fixed << std::setw(5) << std::setprecision(2) << rate << "config/s";
        const auto remaining = rate != 0 ? interval(static_cast<double>(total_ - count_) / rate) : "?";
        std::ostringstream left;
        left.imbue(std::locale::classic());
        left << "UF Pareto: " << std::fixed << std::setw(3) << std::setprecision(0)
             << 100.0 * count_ / total_ << "%|";
        const auto right = "| " + std::to_string(count_) + '/' + std::to_string(total_) +
            " [" + interval(elapsed) + '<' + remaining + ", " + speed.str() +
            (postfix_.empty() ? "" : ", " + postfix_) + ']';
        const auto columns = terminal_columns();
        const auto fixed = display_width(left.str()) + display_width(right);
        const std::size_t width = columns && *columns ? (*columns > fixed ? *columns - fixed : 1) : 10;
        static constexpr std::array<const char*, 9> blocks{" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"};
        const auto units = static_cast<std::size_t>(
            (static_cast<double>(count_) / total_) * width * 8);
        std::string bar;
        for (std::size_t i = 0; i < units / 8; ++i) bar += blocks[8];
        if (units / 8 < width) { bar += blocks[units % 8]; bar.append(width - units / 8 - 1, ' '); }
        auto line = left.str() + bar + right;
        if (columns && !*columns) {
            auto prefix = left.str(); prefix.pop_back();
            line = prefix + right.substr(1);
        } else if (columns) line = trim(line, *columns);
        const auto used = display_width(line);
        write_display("\r" + line + std::string(last_width_ > used ? last_width_ - used : 0, ' '));
        last_width_ = used;
    }
};

}  // namespace uf::pareto_detail
