import { forwardRef, InputHTMLAttributes } from "react";
import clsx from "clsx";

type Props = InputHTMLAttributes<HTMLInputElement>;

export const Input = forwardRef<HTMLInputElement, Props>(function InputBase(
  { className, ...props },
  ref
) {
  return <input ref={ref} className={clsx("input", className)} {...props} />;
});

