export default {
    definition: {
        openapi: '3.1.0',
        info: {
            title: 'ViroGen API Documentation with Swagger',
            version: '0.1.0',
            description:
                'ViroGen : what is the Next Pandemic',
            license: {
                name: 'MIT',
                url: 'https://spdx.org/licenses/MIT.html',
            },
            contact: {
                name: 'ViroGen-Team',
            },
        },
        servers: [
            {
                url: 'http://localhost:3000',
            },
        ],
    },
    apis: ['./src/routes/*.js'],
};