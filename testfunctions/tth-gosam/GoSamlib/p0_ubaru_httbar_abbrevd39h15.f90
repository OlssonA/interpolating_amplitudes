module     p0_ubaru_httbar_abbrevd39h15
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_kinematics, only: epstensor
   use p0_ubaru_httbar_globalsh15
   implicit none
   private
   complex(ki), dimension(0), public :: abb39
   complex(ki), public :: R2d39
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p0_ubaru_httbar_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_color, only: TR
      use p0_ubaru_httbar_globalsl1, only: epspow
      implicit none
      R2d39=0.0_ki
      rat2 = rat2 + R2d39
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='39' value='", &
          & R2d39, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd39h15
