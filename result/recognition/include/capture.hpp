#ifndef CAPTURE_HPP
#define CAPTURE_HPP

#include <vector>
#include <sys/time.h>
#include <lzf.h>
#include <memory>
#include <sys/time.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <semaphore.h>

#ifndef NBUF
#define NBUF 4
#endif

class BufferHandler {
private:
    std::vector<uint8_t> compressed[NBUF], uncompressed[NBUF];
    const uint64_t size;
    uint64_t zip_size[NBUF];
    int compress_pos, writing_pos, copy_pos;
    std::ofstream file;
    sem_t writing, copying, compressing;
    pthread_t write_thread, compress_thread;
    bool is_compressed;
    struct timeval t;

public:
    BufferHandler(const cv::Mat &, const std::string &);
    ~BufferHandler();
    void compress();
    void write();
    void copy(const uchar *);
    void join();
};

void *twrite(void *);

void *tcompress(void *);

#endif /* end of include guard: CAPTURE_HPP */
