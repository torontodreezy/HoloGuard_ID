//
// Created by bladerunner on 10/30/26.
//

#ifndef OPENCV_DATASETLOADER_H
#define OPENCV_DATASETLOADER_H
#include <string>
#include <vector>
struct ClipInfo {
    std::string path;
    std::string category;
    std::string lighting;
    std::string device;
};
class DatasetLoader {
public:
    void loadMetaData(const std::string& directory);
    const std::vector<ClipInfo>& getClips() const;
private:
    std::vector<ClipInfo> clips_;
};


#endif //OPENCV_DATASETLOADER_H
