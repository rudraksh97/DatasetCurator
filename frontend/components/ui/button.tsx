import { ButtonHTMLAttributes } from "react";
import clsx from "clsx";

type Variant = "default" | "secondary" | "ghost";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
}

const variantClass: Record<Variant, string> = {
  default: "btn btn-default",
  secondary: "btn btn-secondary",
  ghost: "btn btn-ghost",
};

export function Button({ className, variant = "default", ...props }: Props) {
  return <button className={clsx(variantClass[variant], className)} {...props} />;
}

