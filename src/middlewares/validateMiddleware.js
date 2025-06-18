const validateMiddleware = (schema) => async (req, res, next) => {
  try {
    await schema.validate({
      body: req.body,
      query: req.query,
      params: req.params,
    });
    return next();
  } catch (error) {
    return res.status(400).json({
      status: "fail",
      message: error.errors.join(", "),
    });
  }
};

export { validateMiddleware };
