module     p0_gg_gh_abbrevd5h0
   use p0_gg_gh_config, only: ki
   use p0_gg_gh_kinematics, only: epstensor
   use p0_gg_gh_globalsh0
   implicit none
   private
   complex(ki), dimension(0), public :: abb5
   complex(ki), public :: R2d5
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p0_gg_gh_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p0_gg_gh_kinematics
      use p0_gg_gh_model
      use p0_gg_gh_color, only: TR
      use p0_gg_gh_globalsl1, only: epspow
      implicit none
      R2d5=0.0_ki
      rat2 = rat2 + R2d5
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='5' value='", &
          & R2d5, "'/>"
      end if
   end subroutine
end module p0_gg_gh_abbrevd5h0
