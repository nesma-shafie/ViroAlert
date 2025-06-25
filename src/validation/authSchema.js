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
    username: usernameField,
    password: passwordField,
  }),
});

const loginSchema = yup.object({
  body: yup.object({
    username: usernameField,
    password:yup.string().required("Password is required") }),
});

export { loginSchema, signupSchema };
