#include "capture.hpp"
#include <liblzf/lzf_c.c>
#include <cassert>

BufferHandler::BufferHandler(const cv::Mat &img, const std::string &filename) : file(filename, std::ios::out | std::ios::binary), size(img.cols * img.rows * img.elemSize()) {
	if (!file.is_open()) {
		std::cout << "Unable to open " << filename << std::endl;
		throw 1;
	}

	file.write(reinterpret_cast<const char *>(&img.cols), sizeof(int));
	file.write(reinterpret_cast<const char *>(&img.rows), sizeof(int));

	int type = img.type();
	file.write(reinterpret_cast<const char *>(&type), sizeof(int));

	size_t elem_size(img.elemSize());
	file.write(reinterpret_cast<const char *>(&elem_size), sizeof(size_t));

	for (int i = 0; i < NBUF; i++) {
		uncompressed[i].resize(size);
		zip_size[i] = size;
	}

	sem_init(&compressing, 0, 0);
    sem_init(&copying, 0, 0);
    sem_init(&writing, 0, 1);

	pthread_create(&compress_thread, NULL, &tcompress, (void *) this);
	pthread_create(&write_thread, NULL, &twrite, (void *) this);
}

BufferHandler::~BufferHandler() {
	sem_destroy(&compressing);
    sem_destroy(&writing);
	sem_destroy(&copying);
}

void BufferHandler::compress() {
	sem_wait(&copying);

	sem_getvalue(&compressing, &compress_pos);
	compress_pos %= NBUF;

	compressed[compress_pos].resize(size + 1000000);
	zip_size[compress_pos] = lzf_compress((const void *) &uncompressed[compress_pos][0], size, (void *) &compressed[compress_pos][0], size);

	sem_post(&compressing);
}

void BufferHandler::write() {
	sem_wait(&compressing);

	sem_getvalue(&writing, &writing_pos);
	writing_pos %= NBUF;

    gettimeofday(&t, NULL);
    file.write(reinterpret_cast<const char *>(&t), sizeof(struct timeval));
	file.write(reinterpret_cast<const char *>(&zip_size[writing_pos]), sizeof(uint64_t));
	if (zip_size[writing_pos] <= 0 || zip_size[writing_pos] > size) {
		for (const auto &value : uncompressed[writing_pos])
			file.write(reinterpret_cast<const char *>(&value), sizeof(value));
	}
	else {
		compressed[writing_pos].resize(zip_size[writing_pos]);
		for (const auto &value : compressed[writing_pos])
			file.write(reinterpret_cast<const char *>(&value), sizeof(value));
	}

	sem_post(&writing);
}

void BufferHandler::copy(const uchar * data) {
	sem_wait(&writing);

	sem_getvalue(&copying, &copy_pos);
	copy_pos %= NBUF;

	memcpy((void *) &uncompressed[copy_pos][0], data, size);

	sem_post(&copying);
}

void BufferHandler::join() {
	sem_wait(&writing);
	pthread_cancel(compress_thread);
	pthread_cancel(write_thread);
}

void *twrite(void *arg) {
	BufferHandler &buf = *((BufferHandler *) arg);

	for (;;) {
		buf.write();
	}
}

void *tcompress(void *arg) {
	BufferHandler &buf = *((BufferHandler *) arg);

	for (;;) {
		buf.compress();
	}
}
