module     p0_gg_gh_abbrevd3h1
   use p0_gg_gh_config, only: ki
   use p0_gg_gh_kinematics, only: epstensor
   use p0_gg_gh_globalsh1
   implicit none
   private
   complex(ki), dimension(12), public :: abb3
   complex(ki), public :: R2d3
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
      abb3(1)=1.0_ki/(mH**2-es23-es12)
      abb3(2)=sqrt(mT**2)
      abb3(3)=sqrt2**(-1)
      abb3(4)=spak2k3**(-1)
      abb3(5)=spbk3k2**(-1)
      abb3(6)=spak1k2**(-1)
      abb3(7)=c1-c2
      abb3(7)=abb3(7)*gHT*i_*e*abb3(5)*abb3(3)*abb3(2)*abb3(1)
      abb3(8)=4.0_ki*spbk3k1**2*abb3(7)*spak2l4*abb3(4)
      abb3(7)=-spbk3k1*abb3(7)
      abb3(9)=abb3(7)*abb3(6)
      abb3(10)=-spbl4k3*spak2l4*abb3(9)
      abb3(11)=abb3(4)*abb3(7)*spak2l4
      abb3(12)=-spbl4k1*abb3(11)
      abb3(10)=2.0_ki*abb3(10)+abb3(12)
      abb3(10)=4.0_ki*abb3(10)
      abb3(9)=16.0_ki*abb3(9)
      abb3(11)=-4.0_ki*spbl4k3*abb3(11)
      abb3(7)=16.0_ki*abb3(7)*abb3(4)
      R2d3=0.0_ki
      rat2 = rat2 + R2d3
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='3' value='", &
          & R2d3, "'/>"
      end if
   end subroutine
end module p0_gg_gh_abbrevd3h1
