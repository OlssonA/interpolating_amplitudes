module     p0_gg_gh_version
   implicit none
   ! The version of Form used for code generation
   integer, parameter, dimension(2) :: formversion = (/4, 3/)
   ! The version of GoSam used for code generation
   integer, parameter, dimension(2) :: gosamversion = (/2, 1/)
   ! The revision of GoSam used for code generation
   integer, parameter :: gosamrevision = int(z'c319210')
end module p0_gg_gh_version
