import pandas as pd
import matplotlib.pyplot as pl

lis = ["1", "2", "3", "4", "5"] #five files to process
versions = ["SharedMem", "Warped", "OneThreadPerEntry", "Dot"]
versionnames = ["Shared Memory", 
                "Warp",
                "Warp, one thread/entry",
                "Dot- Product"]
versionlist=[]

for version in versions:
    dflist = []
    for i, elem in enumerate(lis):
        file_path = 'benchmark/benchmark_'+ version + elem +'.txt'
        column_names = ['N', 't']

        df = pd.read_csv(file_path, delimiter=',', names=column_names, skiprows=11)
        df["N"] = df["N"].astype(int); df["t"] = df["t"].astype(float)
        dflist.append(df)


    concatenated_df = pd.concat(dflist)
    df = concatenated_df.groupby(level=0).mean()# Group by the index and calculate the mean
    df.reset_index(inplace=True)# Reset the index if needed
    versionlist.append(df)

    #print(df)
 
pl.figure(dpi=150); pl.xscale("log")   
for (index, df) in enumerate(versionlist):
    pl.plot(df['N'], df['t'], "o--", label=versionnames[index])
    
    
pl.xlabel("N"); pl.ylabel("Execution time [s]"); pl.legend()
pl.grid(); pl.title("Comparison of execution times, summing operations")
pl.savefig("benchmark/exectimeVectSum.png");pl.show()