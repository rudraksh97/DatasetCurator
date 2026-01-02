import * as React from "react";
import clsx from "clsx";

export interface AvatarProps extends React.HTMLAttributes<HTMLDivElement> {}

export const Avatar = React.forwardRef<HTMLDivElement, AvatarProps>(
  ({ className, children, ...props }, ref) => (
    <div
      ref={ref}
      className={clsx("avatar", className)}
      {...props}
    >
      {children}
    </div>
  )
);
Avatar.displayName = "Avatar";

export const AvatarFallback = React.forwardRef<
  HTMLSpanElement,
  React.HTMLAttributes<HTMLSpanElement>
>(({ className, ...props }, ref) => (
  <span ref={ref} className={clsx("avatar-fallback", className)} {...props} />
));
AvatarFallback.displayName = "AvatarFallback";
