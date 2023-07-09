"""
An example of how to use the AS7262 functions to operate the board.  It begins
with some initial setup, and then puts the board into measurement mode 2, 
which continuously takes measurements for all colours.  This is returned as
a list called "results", which is then printed in a mostly human-readable form.
"""
'''
如何使用AS7262功能操作电路板的示例。 它开始了进行一些初始设置，然后将电路板置于测量模式 2，
连续测量所有颜色。 返回为一个名为“结果”的列表，然后以大部分人类可读的形式打印。
'''
import AS7262_Pi as spec

#重新启动光谱仪，以防万一
spec.soft_reset()

#将器件的增益设置在 0 和 3 之间。 更高的增益 = 更高的读数
spec.set_gain(3)

#将积分时间设置为 1 到 255 之间。 越高意味着读数越长
spec.set_integration_time(50)

#将电路板设置为连续测量所有颜色
spec.set_measurement_mode(2)

#Run this part of the script until stopped with control-C
#运行脚本的这一部分，直到使用 control-C 停止
try:
	#Turn on the main LED
	spec.enable_main_led()
	#Do this until the script is stopped:
	while True:
		#Store the list of readings in the variable "results"
        #将读数列表存储在变量“结果”中
		results = spec.get_calibrated_values()
		#Print the results!
		print("Red    :" + str(results[0]))
		print("Orange :" + str(results[1]))
		print("Yellow :" + str(results[2]))
		print("Green  :" + str(results[3]))
		print("Blue   :" + str(results[4]))
		print("Violet :" + str(results[5]) + "\n")


#When the script is stopped with control-C
except KeyboardInterrupt:
	#Set the board to measure just once (it stops after that)
	spec.set_measurement_mode(3)
	#Turn off the main LED
	spec.disable_main_led()
	#Notify the user
	print("Manually stopped")	
