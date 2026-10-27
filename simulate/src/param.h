#pragma once

#include <iostream>
#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>
#include <filesystem>

namespace param
{

inline struct SimulationConfig
{
    std::string robot;
    std::filesystem::path robot_scene;

    int domain_id;
    std::string interface;

    int use_joystick;
    std::string joystick_type;
    std::string joystick_device;
    int joystick_bits;

    int print_scene_information;

    int enable_elastic_band;
    int band_attached_link = 0;

    // Head camera (optional keys): rendered as the robot's fisheye and published over RTSP.
    int head_camera = 0;
    std::filesystem::path head_camera_model = "head_camera.json";
    std::vector<std::string> head_camera_views = {"head_camera_left", "head_camera_right"};
    // Stereo: one fisheye per eye, one above the other (left eye on top), from these views.
    int head_camera_stereo = 0;
    std::vector<std::string> head_camera_left_eye = {"eye_left_view_left", "eye_left_view_right"};
    std::vector<std::string> head_camera_right_eye = {"eye_right_view_left", "eye_right_view_right"};
    int head_camera_view_size = 640;
    int head_camera_fps = 30;
    std::string head_camera_url = "rtsp://127.0.0.1:8554/robot";
    std::string ffmpeg = "ffmpeg";
    // Depth image of the head's camera module, for the perception process (needs head_camera).
    int depth_camera = 0;
    int depth_camera_width = 544;
    int depth_camera_height = 448;
    int depth_camera_fps = 10;
    std::string depth_camera_shm = "/r1_depth";

    void load_from_yaml(const std::string &filename)
    {
        auto cfg = YAML::LoadFile(filename);
        try
        {
            robot = cfg["robot"].as<std::string>();
            robot_scene = cfg["robot_scene"].as<std::string>();
            domain_id = cfg["domain_id"].as<int>();
            interface = cfg["interface"].as<std::string>();
            use_joystick = cfg["use_joystick"].as<int>();
            joystick_type = cfg["joystick_type"].as<std::string>();
            joystick_device = cfg["joystick_device"].as<std::string>();
            joystick_bits = cfg["joystick_bits"].as<int>();
            print_scene_information = cfg["print_scene_information"].as<int>();
            enable_elastic_band = cfg["enable_elastic_band"].as<int>();
            if (cfg["head_camera"]) head_camera = cfg["head_camera"].as<int>();
            if (cfg["head_camera_model"]) head_camera_model = cfg["head_camera_model"].as<std::string>();
            if (cfg["head_camera_views"]) head_camera_views = cfg["head_camera_views"].as<std::vector<std::string>>();
            if (cfg["head_camera_stereo"]) head_camera_stereo = cfg["head_camera_stereo"].as<int>();
            if (cfg["head_camera_left_eye"]) head_camera_left_eye = cfg["head_camera_left_eye"].as<std::vector<std::string>>();
            if (cfg["head_camera_right_eye"]) head_camera_right_eye = cfg["head_camera_right_eye"].as<std::vector<std::string>>();
            if (cfg["head_camera_view_size"]) head_camera_view_size = cfg["head_camera_view_size"].as<int>();
            if (cfg["head_camera_fps"]) head_camera_fps = cfg["head_camera_fps"].as<int>();
            if (cfg["head_camera_url"]) head_camera_url = cfg["head_camera_url"].as<std::string>();
            if (cfg["ffmpeg"]) ffmpeg = cfg["ffmpeg"].as<std::string>();
            if (cfg["depth_camera"]) depth_camera = cfg["depth_camera"].as<int>();
            if (cfg["depth_camera_width"]) depth_camera_width = cfg["depth_camera_width"].as<int>();
            if (cfg["depth_camera_height"]) depth_camera_height = cfg["depth_camera_height"].as<int>();
            if (cfg["depth_camera_fps"]) depth_camera_fps = cfg["depth_camera_fps"].as<int>();
            if (cfg["depth_camera_shm"]) depth_camera_shm = cfg["depth_camera_shm"].as<std::string>();
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            exit(EXIT_FAILURE);
        }
    }
} config;

/* ---------- Command Line Parameters ---------- */
namespace po = boost::program_options;

//※ This function must be called at the beginning of main() function
inline po::variables_map helper(int argc, char** argv)
{
    po::options_description desc("Unitree Mujoco");
    desc.add_options()
        ("help,h", "Show help message")
        ("domain_id,i", po::value<int>(&config.domain_id), "DDS domain ID; -i 0")
        ("network,n", po::value<std::string>(&config.interface), "DDS network interface; -n eth0")
        ("robot,r", po::value<std::string>(&config.robot), "Robot type; -r go2")
        ("scene,s", po::value<std::filesystem::path>(&config.robot_scene), "Robot scene file; -s scene_terrain.xml")
        ("joystick,j", po::value<int>(&config.use_joystick), "Gamepad as the remote: 1 on, 0 off (overrides use_joystick)")
        ("stereo", po::value<int>(&config.head_camera_stereo), "Head camera: 1 two eyes one above the other, 0 one (overrides head_camera_stereo)")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    
    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        exit(0);
    }

    return vm;
}

}