module     p0_gg_gh_abbrevd9h1
   use p0_gg_gh_config, only: ki
   use p0_gg_gh_kinematics, only: epstensor
   use p0_gg_gh_globalsh1
   implicit none
   private
   complex(ki), dimension(12), public :: abb9
   complex(ki), public :: R2d9
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
      abb9(1)=sqrt(mT**2)
      abb9(2)=sqrt2**(-1)
      abb9(3)=spak2k3**(-1)
      abb9(4)=spbk3k2**(-1)
      abb9(5)=spak1k2**(-1)
      abb9(6)=c1-c2
      abb9(7)=2.0_ki*spbl4k1
      abb9(8)=gHT*i_*e*abb9(4)*abb9(3)*abb9(2)*abb9(1)
      abb9(9)=abb9(8)*spak2l4
      abb9(10)=abb9(9)*spbk3k1
      abb9(11)=abb9(7)*abb9(10)*abb9(6)
      abb9(12)=abb9(6)*abb9(5)
      abb9(10)=-2.0_ki*abb9(10)*abb9(12)
      abb9(9)=abb9(9)*abb9(12)
      abb9(7)=abb9(9)*abb9(7)
      abb9(6)=abb9(6)*abb9(8)
      abb9(8)=-spbk3k1*abb9(6)
      abb9(9)=spbl4k3*abb9(9)
      abb9(8)=2.0_ki*abb9(8)+abb9(9)
      abb9(8)=2.0_ki*abb9(8)
      abb9(6)=8.0_ki*abb9(5)*abb9(6)
      R2d9=0.0_ki
      rat2 = rat2 + R2d9
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='9' value='", &
          & R2d9, "'/>"
      end if
   end subroutine
end module p0_gg_gh_abbrevd9h1
