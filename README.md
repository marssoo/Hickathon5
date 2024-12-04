# Hi! Paris' Hickathon 5 (2024)

Machine Learning competition held at Telecom Paris from November 29 6:00pm to December 1st 4:00pm.

Me and my team finished in 2nd place after exchanging the lead with the one that eventually won for the best part of the week end. Congrats to them, but above all congrats to my team and thanks to you all !

## Team 23

Teams were assigned at random and went from 5 to 7 members. Ours went as follow, in alphabetical order : 

- Massyl Adjal
- Thibaut Boyenval
- Vincent Lagarde
- Marceau Leclerc
- Mila Marsot

## Hi!Paris Hickathon 5 github

[Here](https://github.com/hi-paris/Hickathon5) is the repo. The data we worked on is available at this [link](https://drive.google.com/drive/u/0/folders/1r630CoylbFw7DpnPIYfs1IymY_Nyv3uH).

## Subject

The task was multi-class classification on a dataset elaborated by the Hi!Paris team from several public datasets. The subject was predicting groundwater levels during selected time periods. 

We reached our performance mostly through a thorough pre-processing effort. From there we evaluated a few models, leading us to XGBoost before focusing on finding the right parameters. Our 2nd-place score was achieved by making a "vote" between our 5 best csv files, that were produced by models ranging from 5k  to 15k estimators and with depths of 10 to 13.

The code in this repo was produced fast, hence its packaging. We did not take the time to make something pleasant to read nor optimized for obvious reasons.

