# SNPETask - yolov8
一套qcom rb5下基于SNPE1.61的yolov8 runtime/ts-alg的解耦最小示例，作为对gesanqiu/SNPE_Tutorial的一个补充。基于ultralytics/ultralytics模型，将SiLU替换成ReLU6后重新训练，以适应qcom dsp。由于ultralytics/ultralytics使用AGPL3.0开源协议，此分支也使用AGPL3.0，代码为示例，仅供参考。
请大家多多follow ```@gesanqiu``` !!!
# 编译
```./build.sh```