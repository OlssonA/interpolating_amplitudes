module     p0_ubaru_httbar_abbrevd71h10_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh10_qp
   implicit none
   private
   complex(ki), dimension(40), public :: abb71
   complex(ki), public :: R2d71
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
      abb71(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb71(2)=NC**(-1)
      abb71(3)=es12**(-1)
      abb71(4)=spak2l5**(-1)
      abb71(5)=spbl4k2**(-1)
      abb71(6)=sqrt(mT**2)
      abb71(7)=spak2l3**(-1)
      abb71(8)=spbl3k2**(-1)
      abb71(9)=spbl5k2**(-1)
      abb71(10)=abb71(2)*c1
      abb71(11)=abb71(10)-c2
      abb71(12)=abb71(11)*abb71(2)
      abb71(13)=abb71(12)-c1
      abb71(14)=spak2l3*abb71(4)
      abb71(15)=abb71(14)*spbl3k1
      abb71(16)=abb71(15)+spbl5k1
      abb71(16)=abb71(16)*abb71(13)
      abb71(17)=spbl5k1*NC
      abb71(18)=abb71(15)*NC
      abb71(17)=abb71(17)+abb71(18)
      abb71(17)=abb71(17)*c2
      abb71(16)=abb71(17)+abb71(16)
      abb71(16)=abb71(16)*mT
      abb71(17)=-abb71(2)*spbl5k1*abb71(11)
      abb71(19)=c2*NC
      abb71(20)=abb71(19)-c1
      abb71(21)=-spbl5k1*abb71(20)
      abb71(17)=abb71(17)+abb71(21)
      abb71(21)=abb71(17)*abb71(6)
      abb71(16)=abb71(16)-abb71(21)
      abb71(16)=abb71(16)*spak2l4
      abb71(12)=abb71(12)+abb71(20)
      abb71(22)=spbk2k1*abb71(5)
      abb71(23)=-abb71(12)*abb71(22)*spak2l3*spbl5l3
      abb71(24)=abb71(23)*mT
      abb71(16)=abb71(16)+abb71(24)
      abb71(25)=TR**2*abb71(1)*gs**4*abb71(3)*gHT*e*i_
      abb71(26)=2.0_ki*abb71(25)
      abb71(27)=-abb71(16)*abb71(26)
      abb71(28)=spbl5l3*spal3l5
      abb71(29)=mH**2
      abb71(30)=abb71(29)*abb71(8)
      abb71(31)=abb71(30)*abb71(7)
      abb71(32)=abb71(31)*spbl5k2
      abb71(33)=abb71(32)*spak2l5
      abb71(28)=abb71(28)+abb71(33)
      abb71(28)=abb71(28)*spbl5k1
      abb71(33)=spbk2k1*spbl5l3
      abb71(34)=abb71(30)*abb71(33)
      abb71(35)=abb71(30)*spbl5k2
      abb71(36)=abb71(35)*spbl3k1
      abb71(28)=abb71(28)-abb71(34)+abb71(36)
      abb71(34)=-abb71(28)*abb71(13)
      abb71(28)=-c2*NC*abb71(28)
      abb71(36)=abb71(6)**2
      abb71(37)=2.0_ki*abb71(36)
      abb71(38)=-abb71(17)*abb71(37)
      abb71(28)=abb71(38)+abb71(28)+abb71(34)
      abb71(28)=abb71(6)*abb71(28)
      abb71(34)=mT*abb71(6)
      abb71(38)=abb71(4)*abb71(9)
      abb71(33)=abb71(34)*abb71(12)*abb71(38)*spak2l3*abb71(33)
      abb71(39)=2.0_ki*spbl5k1
      abb71(40)=abb71(39)+abb71(15)
      abb71(40)=abb71(40)*abb71(13)
      abb71(39)=NC*abb71(39)
      abb71(18)=abb71(39)+abb71(18)
      abb71(18)=c2*abb71(18)
      abb71(18)=abb71(18)+abb71(40)
      abb71(18)=abb71(18)*abb71(36)
      abb71(18)=abb71(18)+abb71(33)
      abb71(18)=mT*abb71(18)
      abb71(18)=abb71(28)+abb71(18)
      abb71(18)=spak2l4*abb71(18)
      abb71(28)=abb71(36)+abb71(34)
      abb71(24)=abb71(28)*abb71(24)
      abb71(18)=abb71(24)+abb71(18)
      abb71(18)=abb71(18)*abb71(26)
      abb71(24)=abb71(25)*spak2l4
      abb71(21)=8.0_ki*abb71(24)*abb71(21)
      abb71(15)=spak2l4*abb71(15)*abb71(12)
      abb71(15)=abb71(23)+abb71(15)
      abb71(23)=4.0_ki*abb71(25)
      abb71(15)=abb71(23)*mT*abb71(15)
      abb71(13)=abb71(19)+abb71(13)
      abb71(19)=spbl3k1*abb71(30)*abb71(4)
      abb71(25)=abb71(31)*spbl5k1
      abb71(19)=abb71(19)+abb71(25)
      abb71(13)=4.0_ki*mT*abb71(24)*abb71(19)*abb71(13)
      abb71(16)=abb71(16)*abb71(23)
      abb71(19)=abb71(26)*spak2l4
      abb71(23)=-mT*abb71(19)*abb71(14)*abb71(17)
      abb71(24)=abb71(19)*abb71(6)
      abb71(25)=-abb71(24)*spbl3k1*abb71(12)
      abb71(24)=abb71(24)*spbl5l3*abb71(12)
      abb71(28)=2.0_ki*abb71(34)
      abb71(29)=-abb71(29)-abb71(28)+abb71(37)
      abb71(11)=-abb71(2)*abb71(22)*abb71(11)
      abb71(20)=-abb71(22)*abb71(20)
      abb71(11)=abb71(11)+abb71(20)
      abb71(20)=mT*abb71(11)*abb71(29)
      abb71(10)=abb71(10)*spbk2k1
      abb71(22)=c2*spbk2k1
      abb71(10)=abb71(10)-abb71(22)
      abb71(10)=abb71(10)*abb71(2)
      abb71(22)=abb71(22)*NC
      abb71(29)=c1*spbk2k1
      abb71(10)=abb71(10)+abb71(22)-abb71(29)
      abb71(22)=-abb71(31)*abb71(10)
      abb71(10)=-mT**2*abb71(38)*abb71(10)
      abb71(10)=2.0_ki*abb71(10)+abb71(22)
      abb71(10)=spak2l4*abb71(6)*abb71(10)
      abb71(10)=abb71(20)+abb71(10)
      abb71(10)=abb71(10)*abb71(26)
      abb71(20)=abb71(35)*abb71(11)
      abb71(11)=-abb71(28)*abb71(14)*abb71(11)
      abb71(14)=-spak2l4*abb71(17)*abb71(4)*spal3l5
      abb71(11)=abb71(14)+abb71(11)+abb71(20)
      abb71(11)=abb71(26)*mT*abb71(11)
      abb71(14)=abb71(6)*abb71(32)*abb71(12)
      abb71(12)=-mT*abb71(37)*abb71(4)*abb71(12)
      abb71(12)=abb71(14)+abb71(12)
      abb71(12)=abb71(12)*abb71(19)
      R2d71=abb71(27)
      rat2 = rat2 + R2d71
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='71' value='", &
          & R2d71, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd71h10_qp
