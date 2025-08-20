import app from './app.js';

const PORT = process.env.PORT ;

const server = app.listen(PORT, (err) => {
    if (err) {
        console.error(`Error: ${err}`);
    } else {
        console.log('Success listen on port ', PORT);
    }
});

export default server;