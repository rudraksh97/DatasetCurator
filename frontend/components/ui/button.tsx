import React, { ButtonHTMLAttributes } from "react";
import clsx from "clsx";

type Variant = "default" | "secondary" | "ghost" | "outline";
type Size = "default" | "sm";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  children?: React.ReactNode;
}

const variantClass: Record<Variant, string> = {
  default: "btn btn-default",
  secondary: "btn btn-secondary",
  ghost: "btn btn-ghost",
  outline: "btn btn-outline",
};

const sizeClass: Record<Size, string> = {
  default: "",
  sm: "btn-sm",
};

export function Button({ className, variant = "default", size = "default", ...props }: Props) {
  return <button className={clsx(variantClass[variant], sizeClass[size], className)} {...props} />;
}

