module     p0_ubaru_httbar_abbrevd4h1_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh1_qp
   implicit none
   private
   complex(ki), dimension(32), public :: abb4
   complex(ki), public :: R2d4
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
      abb4(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb4(2)=es12**(-1)
      abb4(3)=spak2l3**(-1)
      abb4(4)=spbl3k2**(-1)
      abb4(5)=spbl5k2**(-1)
      abb4(6)=spak2l4**(-1)
      abb4(7)=spbl4k2**(-1)
      abb4(8)=sqrt(mT**2)
      abb4(9)=NC*c2
      abb4(9)=abb4(9)-c1
      abb4(9)=abb4(9)*i_*e*gHT*abb4(1)*TR**2*gs**4
      abb4(10)=-abb4(2)*abb4(9)
      abb4(11)=-spak1l4*abb4(10)
      abb4(12)=abb4(11)*spal3l5
      abb4(13)=abb4(12)*spbl3k2
      abb4(14)=mT**2
      abb4(15)=-abb4(14)*abb4(11)
      abb4(16)=abb4(11)*abb4(8)
      abb4(17)=-mT*abb4(11)
      abb4(18)=3.0_ki*abb4(16)-abb4(17)
      abb4(18)=abb4(8)*abb4(18)
      abb4(18)=2.0_ki*abb4(15)+abb4(18)
      abb4(18)=spak1l5*abb4(18)
      abb4(19)=2.0_ki*spbl3k2
      abb4(20)=abb4(15)*abb4(19)
      abb4(21)=abb4(17)*abb4(8)
      abb4(22)=abb4(21)*spbl3k2
      abb4(20)=abb4(20)-3.0_ki*abb4(22)
      abb4(23)=spak1l3*abb4(5)
      abb4(20)=abb4(20)*abb4(23)
      abb4(18)=abb4(20)+abb4(18)
      abb4(18)=spbk2k1*abb4(18)
      abb4(20)=abb4(6)*abb4(7)
      abb4(9)=-abb4(14)*spak1l4*abb4(9)*abb4(20)
      abb4(24)=spak1l4**2
      abb4(25)=-abb4(24)*abb4(10)
      abb4(26)=-spbl4k1*abb4(25)
      abb4(9)=abb4(26)+abb4(9)
      abb4(26)=abb4(19)*spal3l5
      abb4(9)=abb4(26)*abb4(9)
      abb4(27)=spak2l5*mH**2*abb4(4)*abb4(3)
      abb4(28)=spbk2k1*abb4(27)
      abb4(29)=spbl3k1*spal3l5
      abb4(28)=abb4(29)+abb4(28)
      abb4(25)=abb4(25)*abb4(28)
      abb4(14)=-abb4(14)*abb4(10)
      abb4(28)=-mT*abb4(10)
      abb4(29)=abb4(28)*abb4(8)
      abb4(30)=abb4(29)+abb4(14)
      abb4(24)=spbk2k1*abb4(5)*abb4(24)*abb4(30)
      abb4(31)=-spak2l4*abb4(13)
      abb4(24)=abb4(31)+abb4(24)+abb4(25)
      abb4(24)=spbl4k2*abb4(24)
      abb4(9)=abb4(24)+abb4(9)+abb4(18)
      abb4(18)=abb4(26)*abb4(11)
      abb4(24)=-4.0_ki*abb4(13)
      abb4(16)=-abb4(16)+abb4(17)
      abb4(16)=abb4(8)*abb4(16)
      abb4(17)=-spbl4k2*abb4(12)
      abb4(15)=abb4(15)+abb4(21)
      abb4(15)=abb4(15)*abb4(5)
      abb4(11)=abb4(27)*abb4(11)
      abb4(11)=abb4(15)-abb4(11)
      abb4(15)=spbl4k2*abb4(11)
      abb4(21)=-abb4(10)*abb4(8)**2
      abb4(21)=abb4(21)-abb4(14)
      abb4(21)=spak1l5*abb4(21)
      abb4(25)=abb4(14)*spbl3k2
      abb4(31)=abb4(29)*spbl3k2
      abb4(32)=-abb4(25)+abb4(31)
      abb4(32)=abb4(32)*abb4(23)
      abb4(20)=-spak1k2*spal3l5*abb4(25)*abb4(20)
      abb4(15)=abb4(20)+abb4(32)+abb4(21)+abb4(15)
      abb4(20)=abb4(5)*abb4(22)
      abb4(21)=2.0_ki*abb4(10)
      abb4(22)=abb4(8)*abb4(21)
      abb4(22)=abb4(22)-abb4(28)
      abb4(22)=abb4(8)*abb4(22)
      abb4(22)=abb4(22)+abb4(14)
      abb4(22)=spal4l5*abb4(22)
      abb4(28)=abb4(19)*abb4(29)
      abb4(25)=-abb4(25)+abb4(28)
      abb4(25)=spal3l4*abb4(5)*abb4(25)
      abb4(22)=abb4(22)+abb4(25)
      abb4(25)=2.0_ki*abb4(7)
      abb4(25)=abb4(25)*abb4(30)
      abb4(28)=-spak1l5*abb4(25)
      abb4(14)=abb4(14)*abb4(19)
      abb4(19)=abb4(14)*abb4(7)
      abb4(23)=-abb4(23)*abb4(19)
      abb4(11)=abb4(23)+abb4(28)-abb4(11)
      abb4(11)=spbk2k1*abb4(11)
      abb4(14)=-abb4(14)+3.0_ki*abb4(31)
      abb4(14)=abb4(14)*abb4(7)*spal3l5
      abb4(23)=spak2l4*abb4(10)*abb4(26)
      abb4(12)=spbl3k1*abb4(12)
      abb4(11)=abb4(12)+abb4(23)+abb4(14)+abb4(11)+2.0_ki*abb4(22)
      abb4(12)=spal3l5*abb4(21)
      abb4(14)=-abb4(5)*abb4(30)
      abb4(10)=abb4(10)*abb4(27)
      abb4(10)=abb4(14)+abb4(10)
      abb4(10)=2.0_ki*abb4(10)
      abb4(14)=-abb4(5)*abb4(19)
      R2d4=abb4(13)
      rat2 = rat2 + R2d4
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='4' value='", &
          & R2d4, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd4h1_qp
