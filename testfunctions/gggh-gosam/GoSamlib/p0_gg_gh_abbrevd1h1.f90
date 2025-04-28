module     p0_gg_gh_abbrevd1h1
   use p0_gg_gh_config, only: ki
   use p0_gg_gh_kinematics, only: epstensor
   use p0_gg_gh_globalsh1
   implicit none
   private
   complex(ki), dimension(13), public :: abb1
   complex(ki), public :: R2d1
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
      abb1(1)=sqrt(mT**2)
      abb1(2)=sqrt2**(-1)
      abb1(3)=es12**(-1)
      abb1(4)=spbk3k2**(-1)
      abb1(5)=spak2k3**(-1)
      abb1(6)=gHT*i_*e*abb1(4)*abb1(3)*abb1(2)*abb1(1)
      abb1(7)=abb1(6)*spak2l4
      abb1(8)=spbk3k1**2
      abb1(9)=abb1(7)*abb1(8)
      abb1(10)=4.0_ki*spbl4k3
      abb1(11)=c1-c2
      abb1(12)=-abb1(10)*abb1(11)*abb1(9)
      abb1(13)=abb1(11)*abb1(5)
      abb1(9)=-4.0_ki*abb1(9)*abb1(13)
      abb1(8)=abb1(8)*abb1(11)*abb1(6)
      abb1(11)=abb1(13)*spbk3k1
      abb1(7)=abb1(7)*abb1(11)
      abb1(13)=-spbl4k1*abb1(7)
      abb1(8)=2.0_ki*abb1(8)+abb1(13)
      abb1(8)=4.0_ki*abb1(8)
      abb1(7)=-abb1(7)*abb1(10)
      abb1(6)=16.0_ki*abb1(6)*abb1(11)
      R2d1=0.0_ki
      rat2 = rat2 + R2d1
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='1' value='", &
          & R2d1, "'/>"
      end if
   end subroutine
end module p0_gg_gh_abbrevd1h1
