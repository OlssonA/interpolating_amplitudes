module     p0_ubaru_httbar_abbrevd2h1_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh1_qp
   implicit none
   private
   complex(ki), dimension(33), public :: abb2
   complex(ki), public :: R2d2
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p0_ubaru_httbar_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_color_qp, only: TR
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      implicit none
      abb2(1)=1.0_ki/(-mT**2+es34)
      abb2(2)=es12**(-1)
      abb2(3)=spak2l3**(-1)
      abb2(4)=spbl3k2**(-1)
      abb2(5)=spbl4k2**(-1)
      abb2(6)=spak2l5**(-1)
      abb2(7)=spbl5k2**(-1)
      abb2(8)=sqrt(mT**2)
      abb2(9)=NC*c2
      abb2(9)=abb2(9)-c1
      abb2(9)=abb2(9)*i_*e*gHT*abb2(1)*TR**2*gs**4
      abb2(10)=-abb2(2)*abb2(9)
      abb2(11)=-spak1l5*abb2(10)
      abb2(12)=abb2(11)*spal3l4
      abb2(13)=abb2(12)*spbl3k2
      abb2(14)=mT**2
      abb2(15)=-abb2(14)*abb2(11)
      abb2(16)=abb2(11)*abb2(8)
      abb2(17)=-mT*abb2(11)
      abb2(18)=3.0_ki*abb2(16)-abb2(17)
      abb2(18)=abb2(8)*abb2(18)
      abb2(18)=2.0_ki*abb2(15)+abb2(18)
      abb2(18)=spak1l4*abb2(18)
      abb2(19)=2.0_ki*spbl3k2
      abb2(20)=abb2(15)*abb2(19)
      abb2(21)=abb2(17)*abb2(8)
      abb2(22)=abb2(21)*spbl3k2
      abb2(20)=abb2(20)-3.0_ki*abb2(22)
      abb2(23)=spak1l3*abb2(5)
      abb2(20)=abb2(20)*abb2(23)
      abb2(18)=abb2(20)+abb2(18)
      abb2(18)=spbk2k1*abb2(18)
      abb2(20)=abb2(6)*abb2(7)
      abb2(9)=-abb2(14)*spak1l5*abb2(9)*abb2(20)
      abb2(24)=spak1l5**2
      abb2(25)=-abb2(24)*abb2(10)
      abb2(26)=-spbl5k1*abb2(25)
      abb2(9)=abb2(26)+abb2(9)
      abb2(26)=abb2(19)*spal3l4
      abb2(9)=abb2(26)*abb2(9)
      abb2(27)=spak2l4*mH**2*abb2(4)*abb2(3)
      abb2(28)=spbk2k1*abb2(27)
      abb2(29)=spbl3k1*spal3l4
      abb2(28)=abb2(29)+abb2(28)
      abb2(25)=abb2(25)*abb2(28)
      abb2(14)=-abb2(14)*abb2(10)
      abb2(28)=-mT*abb2(10)
      abb2(29)=abb2(28)*abb2(8)
      abb2(30)=abb2(29)+abb2(14)
      abb2(24)=spbk2k1*abb2(5)*abb2(24)*abb2(30)
      abb2(31)=-spak2l5*abb2(13)
      abb2(24)=abb2(31)+abb2(24)+abb2(25)
      abb2(24)=spbl5k2*abb2(24)
      abb2(9)=abb2(24)+abb2(9)+abb2(18)
      abb2(18)=abb2(26)*abb2(11)
      abb2(24)=-4.0_ki*abb2(13)
      abb2(25)=-spbl5k2*abb2(12)
      abb2(15)=abb2(15)+abb2(21)
      abb2(15)=abb2(15)*abb2(5)
      abb2(11)=abb2(27)*abb2(11)
      abb2(11)=abb2(15)-abb2(11)
      abb2(15)=spbl5k2*abb2(11)
      abb2(21)=-abb2(10)*abb2(8)**2
      abb2(21)=abb2(21)-abb2(14)
      abb2(21)=spak1l4*abb2(21)
      abb2(31)=abb2(14)*spbl3k2
      abb2(32)=abb2(29)*spbl3k2
      abb2(33)=-abb2(31)+abb2(32)
      abb2(33)=abb2(33)*abb2(23)
      abb2(20)=-spak1k2*spal3l4*abb2(31)*abb2(20)
      abb2(15)=abb2(20)+abb2(33)+abb2(21)+abb2(15)
      abb2(16)=-abb2(16)+abb2(17)
      abb2(16)=abb2(8)*abb2(16)
      abb2(17)=abb2(5)*abb2(22)
      abb2(20)=2.0_ki*abb2(10)
      abb2(21)=-abb2(8)*abb2(20)
      abb2(21)=abb2(21)+abb2(28)
      abb2(21)=abb2(8)*abb2(21)
      abb2(21)=abb2(21)-abb2(14)
      abb2(21)=spal4l5*abb2(21)
      abb2(22)=abb2(19)*abb2(29)
      abb2(22)=-abb2(31)+abb2(22)
      abb2(22)=spal3l5*abb2(5)*abb2(22)
      abb2(21)=abb2(21)+abb2(22)
      abb2(22)=2.0_ki*abb2(7)
      abb2(22)=abb2(22)*abb2(30)
      abb2(28)=-spak1l4*abb2(22)
      abb2(14)=abb2(14)*abb2(19)
      abb2(19)=abb2(14)*abb2(7)
      abb2(23)=-abb2(23)*abb2(19)
      abb2(11)=abb2(23)+abb2(28)-abb2(11)
      abb2(11)=spbk2k1*abb2(11)
      abb2(14)=-abb2(14)+3.0_ki*abb2(32)
      abb2(14)=abb2(14)*abb2(7)*spal3l4
      abb2(23)=spak2l5*abb2(10)*abb2(26)
      abb2(12)=spbl3k1*abb2(12)
      abb2(11)=abb2(12)+abb2(23)+abb2(14)+abb2(11)+2.0_ki*abb2(21)
      abb2(12)=spal3l4*abb2(20)
      abb2(14)=-abb2(5)*abb2(30)
      abb2(10)=abb2(10)*abb2(27)
      abb2(10)=abb2(14)+abb2(10)
      abb2(10)=2.0_ki*abb2(10)
      abb2(14)=-abb2(5)*abb2(19)
      R2d2=abb2(13)
      rat2 = rat2 + R2d2
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='2' value='", &
          & R2d2, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd2h1_qp
