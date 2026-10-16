#pragma once

#include <mln/error.hpp>

namespace Yolo26::Errors {
    namespace Graph::Input::Count {
        inline constexpr Mln::Error Invalid{4300, "Yolo26 expected 1 input, received {}"};
    }

    namespace Preprocess::Image {
        namespace Too {
            inline constexpr Mln::Error Large{
                4301, "Yolo26 received a {}-byte encoded image; the OpenCV limit is {}"};
        }
        namespace Decode {
            inline constexpr Mln::Error Failed{4302, "Yolo26 could not decode the input image"};
        }
    }
}
