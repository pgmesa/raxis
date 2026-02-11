
#include <raxis/core/device.h>

#include <string>
#include <ostream>
#include <stdexcept>


namespace raxis {

Device::Device(const std::string& str) {
    auto colon = str.find(':');
    auto type_str = (colon == std::string::npos) ? str : str.substr(0, colon);

    if (type_str == "cpu")       type_ = DeviceType::CPU;
    else if (type_str == "cuda") type_ = DeviceType::CUDA;
    else throw std::invalid_argument("Unknown device: '" + str + "'");

    if (colon != std::string::npos) {
        index_ = static_cast<DeviceIndex>(std::stoi(str.substr(colon + 1)));
    } else {
        index_ = 0;
    }
}

std::string Device::str() const {
    std::string str = device_type_name(type(), /* lower case */ true);
    if (has_index()) {
        str.push_back(':');
        str.append(std::to_string(index()));
    }
    return str;
}

std::ostream& operator<<(std::ostream& stream, const Device& device) {
  stream << device.str();
  return stream;
}
    
} // namespace raxis
