import * as yup from "yup";
import YupPassword from "yup-password";

YupPassword(yup); // extend yup

const UUIDField = yup.string().required("UUID is required field");

const emailField = yup
  .string()
  .email("must have email format")
  .required("email is required field");

const passwordField = yup
  .string()
  .min(
    8,
    "password must contain 8 or more characters with at least one of each: uppercase, lowercase, number and special"
  )
  .minLowercase(1, "password must contain at least 1 lower case letter")
  .minUppercase(1, "password must contain at least 1 upper case letter")
  .minNumbers(1, "password must contain at least 1 number")
  .minSymbols(1, "password must contain at least 1 special character")
  .required("password is required field");

const usernameField = yup
  .string()
  .min(5, "username must be at least 5 characters")
  .max(191, "username must be at most 191 characters")
  .matches(/^[a-zA-Z0-9_]+$/, "username can only contain letters, numbers,or_")
  .required("username is required field");

const randomBytesTokenField = (name) =>
  yup
    .string()
    .length(8, `${name} must be 8 characters`)
    .required(`${name} is required field`);

export {
  UUIDField,
  emailField,
  passwordField,
  usernameField,
  randomBytesTokenField,
};
