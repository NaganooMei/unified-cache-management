# Lookup 首层预取配置

控制区保存共享索引和预取命令，数据区由 worker 管理。共享缓存实例以 `unique_id` 隔离；`share_buffer_enable=false` 使用进程私有控制区和数据区。

在 scheduler 和 worker 的 Cache 配置中设置：

```yaml
share_buffer_enable: true
cache_prefetch_enable: true       # 默认 false；设为 false 可进行自验证对照
cache_prefetch_batch_size: 32     # 每个 worker 单批上限，范围 1～64
```

仅 `LookupOnPrefix` 的下层命中前缀触发 shard 0 的真实读取。普通 Lookup、反向 Lookup、backend-only 模式不触发该路径。预取不执行 H2D，不占用 Load 保留 slot；锁竞争、缓存占满或命令队列满时允许跳过。

正式 Load 从进入队列到传输完成期间持有 demand 引用，后台预取暂缓提交新批次。已提交的下层读取仍需等待完成，保证 host buffer 的生命周期覆盖异步写入。进程异常退出后需重建共享缓存域，不支持原域内热替换 rank。

开发自验证可在相同 Host Cache 冷态下切换预取开关，观察 DEBUG 日志 `Cache first-layer wait`。该打点位于 shard 0 的下层读取等待前后，不包含 H2D。

Linux 模拟运行时单元测试：

```sh
cmake -S . -B build-cache -DCMAKE_BUILD_TYPE=Debug -DBUILD_UNIT_TESTS=ON -DRUNTIME_ENVIRONMENT=simu
cmake --build build-cache --target ucmstore.test -j4
ctest --test-dir build-cache -R 'CachePrefetchTest|UcmV2CacheBufferTest|UCCacheBufferManagerTest|CacheLoadQueue|CacheDumpQueue' --output-on-failure --timeout 60
```
