import pandas as pd
NumberOfCsvFiles   = 2
dictionary = {}
for  index  in range (NumberOfCsvFiles): 
    dictionary[index] = []

# for each key we acced to onother dict  




def getTargetIndex(ColumnsNamesList):
    target_index = []
    for i in range(len(ColumnsNamesList)):
        index = columns_names.index(ColumnsNamesList[i])
        target_index.append(index)
    return target_index

#ScorrEror Check 

def SCoreErrorCheck(ScoreProRoundList, TotalSCoreList):
    result = 0
    for row, value in enumerate(ScoreProRoundList):
        if row > 0:
            result += value
            print(f"Current row: {row}, Current value: {value}, Current result: {result}, Total score: {TotalSCoreList[row]}")
            if TotalSCoreList[row] != result:
                print(f"Found an error! At index: {row}")
                return True
    return False


def are_floats_approximately_equal(a, b, tolerance=1e-9):
    return abs(a - b) < tolerance


def TimeErrorCheck(ScoreProRoundList, TotalSCoreList):
    result = 0
    for row, value in enumerate(ScoreProRoundList):
        result += value
        print(f"Current row: {row}, Current value: {value}, Current result: {result}, Total score: {TotalSCoreList[row]}")
        if not are_floats_approximately_equal(TotalSCoreList[row], result):
            print(f"Found an error! At index: {row}")
            return True
    return False


def CheckMeanError(step_list, TotalScoreList, MeanScoreList):
    result = 0
    for row, value in enumerate(TotalScoreList):
        result = TotalScoreList[row] / step_list[row]
        print(f"Current row: {row}, Current value: {MeanScoreList[row]}, Current result: {result}")
        if not are_floats_approximately_equal(MeanScoreList[row], result):
            print(f"Found an error! At index: {row}")
            return True
    return False


def CheckTimeOverSCoreError(TotalTimeList, TotalScoreList, TimeOverScoreList):
    for row, value in enumerate(TotalScoreList):
        if TotalScoreList[row] == 0:
            result = TotalTimeList[row]
        else:
            result = TotalTimeList[row] / TotalScoreList[row]
        print(f"Current row: {row}, Current value: {TimeOverScoreList[row]}, Current result: {result}")
        if not are_floats_approximately_equal(TimeOverScoreList[row], result):
            print(f"Found an error! At index: {row}")
            return True
    return False


def CheckTimeOverDeathError(TotalStepsList, TotalTimeList, TimeOverDeathList):
    for row, value in enumerate(TimeOverDeathList):
        result = TotalTimeList[row] / TotalStepsList[row]
        print(f"Current row: {row}, Current value: {TimeOverDeathList[row]}, Current result: {result}")
        if not are_floats_approximately_equal(TimeOverDeathList[row], result):
            print(f"Found an error! At index: {row}")
            return True
    return False

for index  in range (NumberOfCsvFiles): 


    file_path = f"/home/naceur/Desktop/bachelor_project/Project/Multi/Test{index}.csv"

    #we provide the index  

    df = pd.read_csv(file_path)
    columns_names = list(df.columns)
    target_index = getTargetIndex([f"playerScoreProRound{index}", f"playerTotalScore{index}"])
    
    TotalSCoreErrorCheck = SCoreErrorCheck(df[columns_names[target_index[0]]], df[columns_names[target_index[1]]])
    dictionary[index].append(('TotalSCoreErrorCheck',TotalSCoreErrorCheck)) 

    target = getTargetIndex([f"PlayedTimeBeforeDeath{index}", f"TotalPlayedTimeBeforeDeath{index}"])
    TotalTimeErrorCheck = TimeErrorCheck(df[columns_names[target[0]]], df[columns_names[target[1]]])
    dictionary[index].append(('TotalTimeErrorCheck',TotalTimeErrorCheck)) 

    print(f"Columns names: {columns_names}")

    toTarget = getTargetIndex([f"n_games{index}", f"playerTotalScore{index}", f"MeanScore{index}"])
    result = CheckMeanError(df[columns_names[toTarget[0]]], df[columns_names[toTarget[1]]], df[columns_names[toTarget[2]]])
    dictionary[index].append(('result',result ))

    print(f"Result: {result}")

    TargetTotalPlayed = getTargetIndex([f"TotalPlayedTimeBeforeDeath{index}", f"playerTotalScore{index}", f"TimeOverScore{index}"])
    ergebnis = CheckTimeOverSCoreError(
        df[columns_names[TargetTotalPlayed[0]]],
        df[columns_names[TargetTotalPlayed[1]]],
        df[columns_names[TargetTotalPlayed[2]]]
    )
    dictionary[index].append(('ergebnis', ergebnis)) 



    TargetScoreoverDeath = getTargetIndex([f"n_games{index}", f"TotalPlayedTimeBeforeDeath{index}", f"TimeOverDeath{index}"])
    resultaten = CheckTimeOverDeathError(
        df[columns_names[TargetScoreoverDeath[0]]],
        df[columns_names[TargetScoreoverDeath[1]]],
        df[columns_names[TargetScoreoverDeath[2]]]
    )


    dictionary[index].append(('resultaten',resultaten))


    TargetTotalSnakesEaten = getTargetIndex([f"SnakeEatenProRound{index}", f"TotalSnakeEaten{index}"])
    ergebnis = SCoreErrorCheck(
        df[columns_names[TargetTotalSnakesEaten[0]]],
        df[columns_names[TargetTotalSnakesEaten[1]]]
    )
    dictionary[index].append(('TargetTotalSnakesEaten', ergebnis)) 

    
    print(f"Ergebnis: {ergebnis}")

    TargetTotalApplesEaten = getTargetIndex([f"SnakeEatenProRound{index}", f"TotalSnakeEaten{index}"])
    falue = SCoreErrorCheck(
        df[columns_names[TargetTotalApplesEaten[0]]],
        df[columns_names[TargetTotalApplesEaten[1]]]
    )
    dictionary[index].append(('falue',falue))

    print(f"Ergebnis: {falue}")

     
for key, value in dictionary.items():
    print(key, value)
