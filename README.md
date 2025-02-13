# MyQuantAutoEvo
借助大模型自动进化掘金量化策略

# 使用方法
- 掘金量化新建策略,获得策略id和token,填写至config.ini
- 使用OpenAI方案调用大模型,相关配置填写至config.ini
- 执行pip -m install -r requirements.txt安装依赖
- 执行main.py

# 注意
大模型可能生成掘金不存在的方法,导致主线程退出.本项目仅供演示.