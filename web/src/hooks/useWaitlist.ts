import { useMutation } from "@tanstack/react-query";
import { submitWaitlist } from "@/lib/api";

export function useWaitlist() {
  return useMutation({
    mutationFn: ({ name, email }: { name: string; email: string }) =>
      submitWaitlist(name, email),
  });
}
