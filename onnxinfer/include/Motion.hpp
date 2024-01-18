#include "json.hpp"
#include <iostream>
using namespace std;
class Motion
{
public:
    struct Params
    {
        float confidencethreshold;
        float nmsthreshold;
        float objthreshold;
        string modelpath;
        string imgpath;
        bool draw;
        NLOHMANN_DEFINE_TYPE_INTRUSIVE(Params, confidencethreshold, nmsthreshold, objthreshold, modelpath, imgpath, draw); // 添加构造函数
    };
    Params params;
    Motion()
    {
        string jsonPath = "../config/config.json";
        std::ifstream config_is(jsonPath);
        if (!config_is.good())
        {
            std::cout << "Error: Params file path:[" << jsonPath
                      << "] not find .\n";
            exit(-1);
        }

        nlohmann::json js_value;
        config_is >> js_value;
        try
        {
            params = js_value.get<Params>();
        }
        catch (const nlohmann::detail::exception &e)
        {
            std::cerr << "Json Params Parse failed :" << e.what() << '\n';
            exit(-1);
        }
    }
};