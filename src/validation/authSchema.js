import * as yup from "yup";
import YupPassword from "yup-password";
YupPassword(yup); // extend yup

import {
  emailField,
  passwordField,
  UUIDField,
  usernameField,
} from "./fields.js";

const signupSchema = yup.object({
  body: yup.object({
    email: emailField,
    name: yup
      .string()
      .min(3, "name must be at least 3 characters")
      .max(50, "name must be at most 50 characters")
      .required("name is required field"),
    username: usernameField,
    password: passwordField,
  }),
});

const loginSchema = yup.object({
  body: yup.object({
    username: usernameField,
    password: passwordField,
  }),
});

export { loginSchema, signupSchema };
