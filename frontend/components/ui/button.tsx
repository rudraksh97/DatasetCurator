/**
 * Button component with variant and size support.
 */
import React, { ButtonHTMLAttributes } from "react";
import clsx from "clsx";

type Variant = "default" | "secondary" | "ghost" | "outline";
type Size = "default" | "sm";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  /** Visual style variant */
  variant?: Variant;
  /** Size of the button */
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

/**
 * Reusable button component with configurable variants and sizes.
 */
export function Button({ className, variant = "default", size = "default", ...props }: Props) {
  return <button className={clsx(variantClass[variant], sizeClass[size], className)} {...props} />;
}
