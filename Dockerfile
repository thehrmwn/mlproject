# start by pulling the python image
FROM python:3.9-slim

# copy the requirements file into the image
COPY ./requirements.txt /app/requirements.txt

# switch working directory
WORKDIR /usr/src/app

# install the dependencies and packages in the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# copy every content from the local file to the image
COPY . .

# configure the container to run in an executed manner
CMD [ "python", "./application.py" ]


### Build the Docker Image
# docker build -t <name_project> .
# docker build -t student_performance .

### Run the container
# docker run -p 8000:8000 -d student_performance

### Deploying to Docker Hub
## 1. Make a repositories inside DockerHub
## 2. Login to local machine: docker login
## 3. Rename the Docker Image: docker tag student_performance <your-docker-hub-username>/<repository-name>
## 4. Push to DockerHub: docker push <your-docker-hub-username>/<repository-name>