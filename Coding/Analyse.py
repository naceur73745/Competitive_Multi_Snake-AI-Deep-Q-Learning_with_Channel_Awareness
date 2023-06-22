import pandas as pd  
import matplotlib.pyplot as plt


NumberOfSnakes =  5 

for   dataFrameIndex in range(NumberOfSnakes) : 


    file_path = f"/home/naceur/Desktop/bachelor_project/Project/MultiAiSnake/ExcelFiles/Test{dataFrameIndex}.csv"

    df = pd.read_csv(file_path)

    columns_names = list(df.columns)

    print(columns_names)

    # The total Score of each Agent  

    def plot(scores, mean_scores):
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.plot(scores, label='Scores')
        plt.plot(mean_scores, label='Mean Scores')
        plt.ylim(ymin=0)
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
        plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
        plt.legend()
        plt.savefig(f'/home/naceur/Desktop/bachelor_project/Project/MultiAiSnake/images/image{dataFrameIndex}.png')  # Change the path to the desired location
        plt.show(block=False)
        plt.pause(.1)


    # Get the column of Score of the current Snake 

    CurrentPlayerTotalScoredf = df[f'playerTotalScore{dataFrameIndex}']
    CurrentPlayerScoreProRound = df[f'playerScoreProRound{dataFrameIndex}']
    CurrentPlayerTotalScoredfL   =[]
    CurrentPlayerScoreProRoundL  =[]

    for row in  range(len(CurrentPlayerTotalScoredf)) : 
        CurrentPlayerTotalScoredfL.append(df[f'playerTotalScore{dataFrameIndex}'][row])
        CurrentPlayerScoreProRoundL.append( df[f'playerTotalScore{dataFrameIndex}'][row])
        
        plot(CurrentPlayerTotalScoredfL, CurrentPlayerScoreProRoundL)


#we can check other colonnes  etc 

