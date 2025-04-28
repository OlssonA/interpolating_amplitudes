module     p0_ubaru_httbar_abbrevd84h1
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_kinematics, only: epstensor
   use p0_ubaru_httbar_globalsh1
   implicit none
   private
   complex(ki), dimension(15), public :: abb84
   complex(ki), public :: R2d84
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
      abb84(1)=NC**(-1)
      abb84(2)=spak2l5**(-1)
      abb84(3)=spbl5k2**(-1)
      abb84(4)=sqrt(mT**2)
      abb84(5)=2.0_ki*spak1l5
      abb84(6)=TR**2*gHT*e*i_*gs**4
      abb84(5)=abb84(5)*abb84(6)
      abb84(7)=abb84(1)*c1
      abb84(7)=-abb84(7)+2.0_ki*c2
      abb84(8)=abb84(7)*abb84(1)
      abb84(8)=abb84(8)-c1
      abb84(9)=abb84(5)*abb84(8)*spbl4k2*spal4l5
      abb84(10)=abb84(8)*abb84(5)
      abb84(11)=spbl4k2*spak1l4
      abb84(12)=mT*abb84(3)
      abb84(13)=abb84(12)*abb84(4)
      abb84(14)=abb84(8)*abb84(11)*abb84(13)
      abb84(12)=spak2l4*abb84(2)*abb84(12)*spbl4k2
      abb84(12)=abb84(12)+abb84(4)
      abb84(12)=abb84(12)*mT
      abb84(15)=abb84(4)**2
      abb84(12)=abb84(12)-abb84(15)
      abb84(15)=c1*abb84(12)
      abb84(7)=-abb84(1)*abb84(12)*abb84(7)
      abb84(7)=abb84(15)+abb84(7)
      abb84(7)=spak1l5*abb84(7)
      abb84(7)=abb84(7)+abb84(14)
      abb84(7)=2.0_ki*abb84(7)*abb84(6)
      abb84(6)=4.0_ki*abb84(6)
      abb84(12)=spak1l5*abb84(6)*abb84(8)*abb84(2)*abb84(3)*mT**2
      abb84(5)=abb84(5)*abb84(11)*abb84(8)
      abb84(6)=-abb84(6)*abb84(13)*abb84(8)
      R2d84=0.0_ki
      rat2 = rat2 + R2d84
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='84' value='", &
          & R2d84, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd84h1
