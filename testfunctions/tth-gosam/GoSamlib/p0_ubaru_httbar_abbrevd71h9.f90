module     p0_ubaru_httbar_abbrevd71h9
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_kinematics, only: epstensor
   use p0_ubaru_httbar_globalsh9
   implicit none
   private
   complex(ki), dimension(22), public :: abb71
   complex(ki), public :: R2d71
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
      abb71(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb71(2)=NC**(-1)
      abb71(3)=es12**(-1)
      abb71(4)=spak2l5**(-1)
      abb71(5)=sqrt(mT**2)
      abb71(6)=spak2l3**(-1)
      abb71(7)=spbl3k2**(-1)
      abb71(8)=abb71(2)*c1
      abb71(8)=abb71(8)-c2
      abb71(9)=abb71(8)*abb71(2)
      abb71(10)=c2*NC
      abb71(10)=abb71(10)-c1
      abb71(9)=abb71(9)+abb71(10)
      abb71(11)=abb71(9)*mT
      abb71(12)=abb71(9)*abb71(5)
      abb71(11)=abb71(11)+abb71(12)
      abb71(13)=abb71(11)*spbl5k2
      abb71(14)=abb71(4)*spak2l3
      abb71(15)=-mT*abb71(9)*abb71(14)*spbl3k2
      abb71(13)=abb71(13)-abb71(15)
      abb71(16)=abb71(3)*TR**2*abb71(1)*gs**4*gHT*e*spak1l4*i_
      abb71(17)=2.0_ki*abb71(16)
      abb71(18)=-abb71(13)*abb71(17)
      abb71(19)=spbl5l3*spal3l5
      abb71(20)=mH**2
      abb71(19)=abb71(19)+abb71(20)
      abb71(19)=-abb71(19)*abb71(9)
      abb71(11)=abb71(5)*abb71(11)
      abb71(21)=abb71(20)*abb71(6)*abb71(7)
      abb71(22)=-spbl5k2*abb71(9)*abb71(21)*spak2l5
      abb71(11)=abb71(22)+2.0_ki*abb71(11)+abb71(19)
      abb71(11)=spbl5k2*abb71(5)*abb71(11)
      abb71(19)=abb71(5)**2
      abb71(22)=-abb71(19)*abb71(15)
      abb71(11)=abb71(22)+abb71(11)
      abb71(11)=abb71(11)*abb71(17)
      abb71(12)=-8.0_ki*abb71(12)*abb71(16)*spbl5k2
      abb71(16)=4.0_ki*abb71(16)
      abb71(15)=-abb71(15)*abb71(16)
      abb71(19)=-2.0_ki*abb71(19)+abb71(20)
      abb71(8)=abb71(2)*abb71(4)*abb71(8)
      abb71(10)=abb71(4)*abb71(10)
      abb71(8)=abb71(8)+abb71(10)
      abb71(10)=mT*abb71(8)*abb71(19)
      abb71(19)=-mT-abb71(5)
      abb71(19)=-spbl5k2*abb71(19)*abb71(21)*abb71(9)
      abb71(10)=abb71(19)+abb71(10)
      abb71(10)=abb71(10)*abb71(16)
      abb71(13)=abb71(13)*abb71(16)
      abb71(16)=mT*abb71(17)*spbl5k2
      abb71(14)=abb71(16)*abb71(14)*abb71(9)
      abb71(17)=abb71(17)*abb71(5)
      abb71(19)=-abb71(17)*spbl3k2*abb71(9)
      abb71(9)=abb71(17)*spbl5l3*abb71(9)
      abb71(8)=abb71(16)*spal3l5*abb71(8)
      R2d71=abb71(18)
      rat2 = rat2 + R2d71
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='71' value='", &
          & R2d71, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd71h9
