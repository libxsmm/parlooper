import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_lines(filePath):
    with open(filePath) as f:
        for line in f:
            line = line.strip() #removes newline character from end
            yield line

filename = sys.argv[1]
gen = generate_lines(filename)
data_full = np.genfromtxt(gen, delimiter=' ', dtype=None, encoding=None)

filename2 = sys.argv[2]
gen2 = generate_lines(filename2)
data2_full = np.genfromtxt(gen2, delimiter=' ', dtype=None, encoding=None)

gemm_name = sys.argv[3]

data = [elem[0] for elem in data_full]
data2 = [elem[0] for elem in data2_full]

fig,ax = plt.subplots()
lns1 = ax.plot(data,label='Measured performance in GFLOPS')
ax2 = ax.twinx()
lns2 = ax2.plot(data2,label='Model score', color='red')
sorted_vals = sorted(set(data2), reverse=True)
max1 = max(data)
predicted_max = data[0]
max2 = max(data2)
maxall = max(max1, max2)
data3 = [0,0]
lns3 = ax.plot(data3,label='Top-5 performant model classes', color='yellow')
count = 0
i=0
while (data2[i] >= sorted_vals[4]):
    count += 1
    i+=1
v_list = list(range(1, count))
ax.vlines(v_list, 0, maxall, linestyles='solid', colors='yellow', label='Top-5 performant model classes')
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]

#xarrow_loc = len(data)/2
#percentage = round((max1-predicted_max)/max1*100)
#ax.annotate("  " + str(percentage) + " % gap",  xy=(xarrow_loc, predicted_max+(max1-predicted_max)/2), xytext=(xarrow_loc, predicted_max+(max1-predicted_max)/2), arrowprops={"arrowstyle":"-|>", "color":"white"} )
#ax.annotate("", xy=(xarrow_loc, predicted_max), xytext=(xarrow_loc, max1), arrowprops={"arrowstyle":"-|>", "color":"green"})
#ax.annotate("", xy=(xarrow_loc, predicted_max), xytext=(xarrow_loc, max1), arrowprops={"arrowstyle":"<|-", "color":"green"})
#plt.axhline(y=max1, color='green', linestyle='dotted')
#plt.axhline(y=predicted_max, color='green', linestyle='dotted')

#ax.legend(lns3.get_label())
#ax.legend()
#ax2.legend()
ax.set_xlabel("Loop schedules")
ax.set_ylabel(r"Measured performance in GFLOPS")
ax2.set_ylabel(r"Model score ")
plt.legend(lns,labs,facecolor='white', framealpha=1)
plt.title("GEMM " + gemm_name)
plt.savefig(gemm_name+".pdf", bbox_inches='tight')

