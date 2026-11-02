To build a docker image:

launch docker desktop
open up powershell with administrator rights
The Dockerfile is in the config folder, so from the root of the repository, run:
docker build -f config/Dockerfile .
