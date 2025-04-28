module     p0_gg_gh_abbrevd7h1
   use p0_gg_gh_config, only: ki
   use p0_gg_gh_kinematics, only: epstensor
   use p0_gg_gh_globalsh1
   implicit none
   private
   complex(ki), dimension(9), public :: abb7
   complex(ki), public :: R2d7
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
      abb7(1)=sqrt(mT**2)
      abb7(2)=sqrt2**(-1)
      abb7(3)=spbk3k2**(-1)
      abb7(4)=spak2k3**(-1)
      abb7(5)=spak1k2**(-1)
      abb7(6)=c2-c1
      abb7(7)=abb7(1)*abb7(2)*abb7(3)*gHT*e*i_
      abb7(8)=2.0_ki*spbk3k1**2*abb7(6)*abb7(7)
      abb7(9)=8.0_ki*abb7(4)
      abb7(6)=abb7(6)*abb7(9)*abb7(7)
      abb7(7)=spbk3k1*abb7(6)
      abb7(6)=-abb7(5)*abb7(6)
      R2d7=0.0_ki
      rat2 = rat2 + R2d7
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='7' value='", &
          & R2d7, "'/>"
      end if
   end subroutine
end module p0_gg_gh_abbrevd7h1
