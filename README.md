# BlogDB

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

3. **Add your MongoDB URL to an Heroku config var**:

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
