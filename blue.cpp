/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "machine/runtime/memory_utils/memory_pool.h"
#include <iomanip>
#include <sstream>
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error_code.h"
#include "adapter/api/runtime_api.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {
namespace {
inline constexpr int RTMALLOC_SUCCESS = 0;
inline constexpr size_t ONT_GB_SIZE = 1024 * 1024 * 1024;
inline constexpr uint64_t SENTINEL_VALUE = 0xDEADBEEFDEADBEEF;
inline constexpr uint32_t SENTINEL_NUM = 64;
inline constexpr uint32_t SENTINEL_MEM_SIZE = 512;
inline uint64_t MemSizeAlign(const uint64_t bytes, const uint32_t aligns = 512U)
{
    const uint64_t alignSize = (aligns == 0U) ? sizeof(uintptr_t) : aligns;
    return (((bytes + alignSize) - 1U) / alignSize) * alignSize;
}
}

MemoryBlock::MemoryBlock(void* addr, size_t size, bool is_huge)
    : base_addr(addr), block_size(size), used_size(0), is_huge_1g(is_huge)
{
    Init();
}

void MemoryBlock::Init()
{
    if (is_huge_1g) {
        free_map[reinterpret_cast<uintptr_t>(base_addr)] = block_size;
    } else {
        free_map.clear();
    }
}

void* MemoryBlock::Allocate(uint64_t alignSize)
{
    if (!is_huge_1g) {
        if (used_size == 0 && block_size >= alignSize) {
            used_size = block_size;
            return base_addr;
        } else {
            int32_t devId = -1;
            RuntimeGetDevice(&devId);
            MACHINE_LOGE(
                DevCommonErr::ALLOC_FAILED,
                "[MemPool] 2MB block alloc failed on device=%d. used_size=%zu (%.2f MB), block_size=%zu (%.2f MB), "
                "req=%lu (%.2f MB), available=%zu (%.2f MB)",
                devId, used_size, used_size / 1048576.0, block_size, block_size / 1048576.0,
                alignSize, alignSize / 1048576.0,
                block_size - used_size, (block_size - used_size) / 1048576.0);
            return nullptr;
        }
    }

    for (auto it = free_map.begin(); it != free_map.end(); ++it) {
        uintptr_t chunk_addr = it->first;
        size_t chunk_size = it->second;

        if (chunk_size >= alignSize) {
            void* use_ptr = reinterpret_cast<void*>(chunk_addr);
            size_t remaining = chunk_size - alignSize;

            free_map.erase(it);

            if (remaining > 0) {
                free_map[chunk_addr + alignSize] = remaining;
            }

            used_size += alignSize;
            MACHINE_LOGI(
                "Allocate in 1GB block: ptr=%p, chunkSize=%zu, alignSize=%lu.", use_ptr, chunk_size, alignSize);
            return use_ptr;
        }
    }
    return nullptr;
}

void MemoryBlock::Free(void* ptr, size_t size)
{
    if (!is_huge_1g) {
        MACHINE_LOGE(DevCommonErr::FREE_FAILED, "Logic Error: 2MB block should not call Free()");
        return;
    }

    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);

    free_map[addr] = size;
    used_size -= size;

    auto it = free_map.find(addr);
    if (it == free_map.end())
        return;

    auto next_it = std::next(it);
    if (next_it != free_map.end()) {
        if (it->first + it->second == next_it->first) {
            it->second += next_it->second;
            free_map.erase(next_it);
        }
    }

    if (it != free_map.begin()) {
        auto prev_it = std::prev(it);
        if (prev_it->first + prev_it->second == it->first) {
            prev_it->second += it->second;
            free_map.erase(it);
        }
    }
}

DevMemoryPool& DevMemoryPool::Instance()
{
    static DevMemoryPool memoryPool;
    return memoryPool;
}

DevMemoryPool::DevMemoryPool()
{
    needMemCheck_ = (config::GetDebugOption<int64_t>(CFG_RUNTIME_DBEUG_MODE) == CFG_DEBUG_ALL);
    sentinelVec_ = std::vector<uint64_t>(SENTINEL_NUM, SENTINEL_VALUE);
}

DevMemoryPool::~DevMemoryPool()
{
    CheckAllSentinels();
    DestroyPool();
}

void DevMemoryPool::AllocDevAddr(uint8_t** devAddr, const uint64_t size)
{
    if (!AllocDevAddrInPool(devAddr, size)) {
        int32_t devId = -1;
        RuntimeGetDevice(&devId);
        MACHINE_LOGE(DevCommonErr::ALLOC_FAILED,
            "[MemPool] AllocDevAddr FAILED on device=%d. size=%lu (%.2f MB), caller=%p",
            devId, size, size / 1048576.0, __builtin_return_address(0));
        devAddr = nullptr;
    } else {
        MACHINE_LOGI("RuntimeAgentMemory: Alloc success %p", *devAddr);
    }
}

bool DevMemoryPool::AllocDevAddrInPool(uint8_t** devAddr, const uint64_t size)
{
    if (size == 0)
        return false;
    if (devAddr == nullptr) {
        MACHINE_LOGE(DevCommonErr::NULLPTR, "devAddr is nullptr");
        return false;
    }
    auto alignSize = MemSizeAlign(size);
    if (needMemCheck_) {
        alignSize += SENTINEL_MEM_SIZE;
    }

    for (auto& block : memoryBlocks_) {
        void* ptr = block->Allocate(alignSize);
        if (ptr != nullptr) {
            *devAddr = static_cast<uint8_t*>(ptr);
            RecordAllocation(ptr, block, alignSize);
            PutSentinelAddr(*devAddr, size);
            return true;
        }
    }

    MemoryBlock* newBlock = CreateNewBlock(alignSize);
    if (newBlock != nullptr) {
        void* ptr = newBlock->Allocate(alignSize);
        if (ptr != nullptr) {
            *devAddr = static_cast<uint8_t*>(ptr);
            RecordAllocation(ptr, newBlock, alignSize);
            PutSentinelAddr(*devAddr, size);
            return true;
        }
    }

    int32_t devId = -1;
    RuntimeGetDevice(&devId);
    size_t poolTotal = 0, poolUsed = 0, poolFree = 0;
    for (const auto& blk : memoryBlocks_) {
        poolTotal += blk->block_size;
        poolUsed += blk->used_size;
    }
    poolFree = poolTotal - poolUsed;
    MACHINE_LOGE(DevCommonErr::ALLOC_FAILED,
        "[MemPool] AllocDevAddrInPool FAILED on device=%d. "
        "req_size=%lu (%.2f MB), aligned_size=%lu (%.2f MB), "
        "pool_total=%zu (%.2f MB), pool_used=%zu (%.2f MB), pool_free=%zu (%.2f MB), "
        "num_blocks=%zu, caller=%p",
        devId, size, size / 1048576.0, alignSize, alignSize / 1048576.0,
        poolTotal, poolTotal / 1048576.0, poolUsed, poolUsed / 1048576.0,
        poolFree, poolFree / 1048576.0, memoryBlocks_.size(),
        __builtin_return_address(0));
    PrintPoolStatus();
    return false;
}

void DevMemoryPool::FreeDevAddr(void* ptr)
{
    if (ptr == nullptr) {
        MACHINE_LOGE(DevCommonErr::NULLPTR, "Freeing nullptr");
        return;
    }
    CheckSentinel(static_cast<uint8_t*>(ptr), true);

    auto it = addrToBlock_.find(ptr);
    if (it == addrToBlock_.end()) {
        MACHINE_LOGE(DevCommonErr::FREE_FAILED, "Freeing unknown pointer: %p", ptr);
        return;
    }

    MemoryBlock* block = it->second;
    size_t size = allocSizes_[ptr];

    if (block->is_huge_1g) {
        block->Free(ptr, size);
    } else {
        MACHINE_LOGI("Directly freeing 2MB block: addr=%p.", block->base_addr);
        FreeMemBlock(block);
        for (auto vec_it = memoryBlocks_.begin(); vec_it != memoryBlocks_.end(); ++vec_it) {
            if (*vec_it == block) {
                memoryBlocks_.erase(vec_it);
                break;
            }
        }
    }

    addrToBlock_.erase(it);
    allocSizes_.erase(ptr);
}

void DevMemoryPool::PutSentinelAddr(uint8_t* baseAddr, uint64_t baseSize)
{
    if (needMemCheck_) {
        uint8_t* sentinelAddr = baseAddr + baseSize;
        if (RuntimeMemcpy(sentinelAddr, SENTINEL_MEM_SIZE, sentinelVec_.data(), SENTINEL_MEM_SIZE,
                          RtMemcpyKind::HOST_TO_DEVICE) != 0) {
            MACHINE_LOGW("Memory copy sentinel value failed! Do not check memory.");
            return;
        }
        MACHINE_LOGI("Base addr add: baseAddr=%p, sentinelAddr=%p.", baseAddr, sentinelAddr);
        sentinelValMap_[baseAddr].push_back(sentinelAddr);
    }
}

bool DevMemoryPool::CheckAllSentinels()
{
    if (!needMemCheck_) {
        return true;
    }
    bool allGood = true;
    for (auto& iter : sentinelValMap_) {
        if (!CheckSentinel(iter.first, false)) {
            allGood = false;
        }
    }
    if (!allGood) {
        MACHINE_LOGE(HostLauncherErr::MEM_POOL_CHECK_ALL_SENTINELS_FAILED, "CheckAllSentinels failed.");
    }
    sentinelValMap_.clear();
    return allGood;
}

void DevMemoryPool::PrintSentinelVal(std::vector<uint64_t>& sentinelVal, uint8_t* sentinelAddr)
{
    std::ostringstream oss;
    uint8_t* byte_ptr = reinterpret_cast<uint8_t*>(sentinelVal.data());
    oss << "Print Sentinel val in hex with ori val[" << std::hex << "0x" << SENTINEL_VALUE << "]" << std::endl;
    MACHINE_LOGW("%s", oss.str().c_str());
    oss.str("");
    for (uint32_t i = 0; i < SENTINEL_MEM_SIZE; ++i) {
        oss << std::hex << std::setw(2) << std::setfill('0') << (int)byte_ptr[i];
        if ((i + 1) % 16 == 0) {
            oss << std::endl;
        } else {
            oss << " ";
        }
        if ((i + 1) % 64 == 0) {
            MACHINE_LOGW("Sentinel Addr:%p Val:[\n%s]", sentinelAddr + i, oss.str().c_str());
            oss.str("");
        }
    }
}

bool DevMemoryPool::CheckSentinel(uint8_t* baseAddr, bool remove)
{
    if (!needMemCheck_ || sentinelValMap_.empty()) {
        return true;
    }
    if (baseAddr == reinterpret_cast<uint8_t*>(0x12345678)) {
        return true;
    }
    auto iter = sentinelValMap_.find(baseAddr);
    if (iter == sentinelValMap_.end()) {
        MACHINE_LOGE(DevCommonErr::PARAM_CHECK_FAILED, "Base addr %p not found in map, need check code.", baseAddr);
        return false;
    }
    std::vector<uint64_t> sentinelVal(SENTINEL_NUM, 0);
    bool allGood = true;
    auto& sentinelVec = iter->second;
    for (auto sentinelAddr : sentinelVec) {
        MACHINE_LOGI("Check Sentinel: baseAddr=%p, sentinelAddr=%p.", baseAddr, sentinelAddr);
        if (RuntimeMemcpy(sentinelVal.data(), SENTINEL_MEM_SIZE, sentinelAddr, SENTINEL_MEM_SIZE,
            RtMemcpyKind::DEVICE_TO_HOST) != 0) {
            MACHINE_LOGW("Memory copy D2H failed! Do not check memory.");
            break;
        }
        if (memcmp(sentinelVal.data(), sentinelVec_.data(), SENTINEL_MEM_SIZE) != 0) {
            PrintSentinelVal(sentinelVal, sentinelAddr);
            allGood = false;
        }
    }
    if (!allGood) {
        MACHINE_LOGE(DevCommonErr::PARAM_CHECK_FAILED, "BaseAddr:%p check sentinel failed.", baseAddr);
    } else {
        MACHINE_LOGI("BaseAddr:%p check sentinel Ok.", baseAddr);
    }
    if (remove) {
        sentinelValMap_.erase(baseAddr);
    }
    return allGood;
}

void DevMemoryPool::DynamicRecycle()
{
    auto it = memoryBlocks_.begin();
    while (it != memoryBlocks_.end()) {
        if ((*it)->used_size == 0) {
            MACHINE_LOGI("Recycling empty block: addr=%p", (*it)->base_addr);
            FreeMemBlock(*it);
            it = memoryBlocks_.erase(it);
        } else {
            ++it;
        }
    }
}

void DevMemoryPool::DestroyPool()
{
    for (auto& block : memoryBlocks_) {
        if (block != nullptr) {
            FreeMemBlock(block);
        }
    }
    memoryBlocks_.clear();
    addrToBlock_.clear();
    allocSizes_.clear();
    MACHINE_LOGI("MemPool destroyed, all memory freed");
}

void DevMemoryPool::PrintPoolStatus() const
{
    size_t cnt_1g = 0;
    size_t cnt_2m = 0;
    size_t total = 0;
    size_t used = 0;
    MACHINE_LOGI("========== [Memory Pool Status] ==========");
    for (size_t i = 0; i < memoryBlocks_.size(); ++i) {
        auto* blk = memoryBlocks_[i];
        if (blk->is_huge_1g)
            cnt_1g++;
        else
            cnt_2m++;
        total += blk->block_size;
        used += blk->used_size;

        double rate = blk->block_size ? (double)blk->used_size * 100.0 / blk->block_size : 0;
        MACHINE_LOGI(
            "Block[%lu] %s | Addr: %p | Used: %.1f%% | Fragments: %lu", i, blk->is_huge_1g ? "1G" : "2M",
            blk->base_addr, rate, blk->free_map.size());
    }
    MACHINE_LOGI("Summary: 1G x %lu, 2M x %lu | Used/Total: %lu/%lu MB", cnt_1g, cnt_2m, used >> 20, total >> 20);
}

void DevMemoryPool::FreeMemBlock(MemoryBlock* block)
{
    if (block == nullptr) {
        return;
    }

    if (block->base_addr != nullptr) {
        MACHINE_LOGI("Releasing physical memory: addr=%p, size=%lu", block->base_addr, block->block_size);
        RuntimeFree(block->base_addr);
        block->base_addr = nullptr;
    }
    delete block;
    block = nullptr;
}

void DevMemoryPool::RecordAllocation(void* ptr, MemoryBlock* block, size_t size)
{
    addrToBlock_[ptr] = block;
    allocSizes_[ptr] = size;
}

MemoryBlock* DevMemoryPool::CreateNewBlock(uint64_t alignSize)
{
    uint8_t* devAddr = nullptr;
    size_t size1G = ((alignSize - 1) / ONT_GB_SIZE + 1) * ONT_GB_SIZE;

    int32_t devId = -1;
    RuntimeGetDevice(&devId);

    auto rc1 = RuntimeMalloc((void**)&devAddr, size1G, ONG_GB_HUGE_PAGE_FLAGS, 0);
    if (rc1 == RTMALLOC_SUCCESS) {
        MACHINE_LOGI("[MemPool] device=%d: allocated 1GB huge page block, size=%zu (%.2f MB)",
                     devId, size1G, size1G / 1048576.0);
        MemoryBlock* block = new MemoryBlock(devAddr, size1G, true);
        memoryBlocks_.push_back(block);
        return block;
    }
    MACHINE_LOGW("[MemPool] device=%d: 1GB huge page alloc FAILED (rc=%d, req=%zu / %.2f MB)",
                 devId, rc1, size1G, size1G / 1048576.0);

    devAddr = nullptr;
    auto rc2 = RuntimeMalloc((void**)&devAddr, alignSize, TWO_MB_HUGE_PAGE_FLAGS, 0);
    if (rc2 == RTMALLOC_SUCCESS) {
        MACHINE_LOGI("[MemPool] device=%d: allocated 2MB page block, size=%lu (%.2f MB)",
                     devId, alignSize, alignSize / 1048576.0);
        MemoryBlock* block = new MemoryBlock(devAddr, alignSize, false);
        memoryBlocks_.push_back(block);
        return block;
    }
    MACHINE_LOGW("[MemPool] device=%d: 2MB page alloc FAILED (rc=%d, req=%lu / %.2f MB)",
                 devId, rc2, alignSize, alignSize / 1048576.0);

    MACHINE_LOGE(DevCommonErr::ALLOC_FAILED,
        "[MemPool] All memory alloc strategies failed on device=%d. "
        "alignSize=%lu (%.2f MB), 1GB_attempt=%zu (%.2f MB, rc=%d), "
        "2MB_attempt=%lu (%.2f MB, rc=%d)",
        devId, alignSize, alignSize / 1048576.0,
        size1G, size1G / 1048576.0, rc1,
        alignSize, alignSize / 1048576.0, rc2);
    return nullptr;
}
} // namespace npu::tile_fwk
