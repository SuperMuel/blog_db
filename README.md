# [BlogDB](https://github.com/SuperMuel/blog_db)

## Introduction

## Installation

### Prerequisites

1. **Python 3.12**: Install Python from [here](https://www.python.org/downloads/).
2. **Poetry**: Install Poetry from [here](https://python-poetry.org/docs/).
3. **A MongoDB database**: You can create a free MongoDB database at [MongoDB Atlas](https://www.mongodb.com/cloud/atlas).

### Local Development

1. **Clone the repository**:

```sh
git clone https://github.com/SuperMuel/blog_db
```

2. **Install dependencies**:

```sh
poetry install
```

3. **Create a `.env` file**:

Copy the contents of the `.env.example` file to a new file called `.env` and fill it accordingly.

4. **Serve the API**:

```sh
poetry run fastapi dev main.py
```

The API will be available at `http://localhost:8000`.

You can check the documentation at `http://localhost:8000/docs`.

## Deploy to Heroku

### Prerequisites

1. **Heroku account**: Sign up at [Heroku](https://www.heroku.com/).
2. **Heroku CLI**: Install the Heroku CLI from [here](https://devcenter.heroku.com/articles/heroku-cli).

### Steps

1. **Login to Heroku CLI**:

```bash
heroku login
```

2. **Create a new Heroku app**:

```sh
heroku create your-app-name
```

3. **Add your MongoDB URI to an Heroku config var**:

```sh
heroku config:set MONGODB_URI=your-mongodb-uri
```

4. **Add the Poetry buildpack**:

```sh
heroku buildpacks:clear
heroku buildpacks:add https://github.com/moneymeets/python-poetry-buildpack.git
heroku buildpacks:add heroku/python

```

5. **Deploy the app**:

```sh
git push heroku main
```

6. **Scale the app**:

This will start a web dyno to serve the application.

```sh
heroku ps:scale web=1
```

### Additional Tips

Logging: Use Heroku logs to debug any issues with your app:

```sh
heroku logs --tail
```
