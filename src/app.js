import express from "express";
import swaggerConfig from "./swaggerConfig.js";
import swaggerJsdoc from "swagger-jsdoc";
import swaggerUi from "swagger-ui-express";

import authRouter from "./routes/authRoutes.js";
import userRouter from "./routes/userRoutes.js";

// config swagger
const swaggerSpecs = swaggerJsdoc(swaggerConfig);
const app = express();

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// swagger documentation route
app.use(
    '/ViroGen/app/api-docs',
    swaggerUi.serve,
    swaggerUi.setup(swaggerSpecs, { explorer: true })
);

// our main routes
app.use("/ViroGen/app/auth", authRouter);
app.use("/ViroGen/app/user", userRouter);

// handle all other routes
app.all("*", (req, res) => {
  return res.status(404).json({
    status: "fail",
    message: `Can't find ${req.originalUrl} on this server!`,
  });
});

export default app;
