module     p0_ubaru_httbar_abbrevd3h5_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh5_qp
   implicit none
   private
   complex(ki), dimension(33), public :: abb3
   complex(ki), public :: R2d3
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
      abb3(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb3(2)=NC**(-1)
      abb3(3)=es12**(-1)
      abb3(4)=spbl5k2**(-1)
      abb3(5)=spak2l4**(-1)
      abb3(6)=sqrt(mT**2)
      abb3(7)=spak2l3**(-1)
      abb3(8)=spbl3k2**(-1)
      abb3(9)=i_*e*gHT*abb3(1)*TR**2*gs**4
      abb3(10)=abb3(9)*abb3(3)
      abb3(11)=abb3(10)*abb3(6)
      abb3(12)=abb3(2)**2
      abb3(13)=abb3(11)*abb3(12)
      abb3(14)=abb3(12)*abb3(9)
      abb3(15)=abb3(3)*abb3(14)*mT
      abb3(13)=abb3(13)+abb3(15)
      abb3(13)=abb3(13)*c1
      abb3(16)=mT*abb3(2)
      abb3(17)=abb3(10)*abb3(16)
      abb3(18)=abb3(11)*abb3(2)
      abb3(19)=abb3(18)+abb3(17)
      abb3(19)=abb3(19)*c2
      abb3(13)=abb3(13)-abb3(19)
      abb3(19)=abb3(13)*spbl4k2
      abb3(20)=abb3(19)*spak1l5
      abb3(21)=abb3(15)*c1
      abb3(17)=abb3(17)*c2
      abb3(17)=abb3(21)-abb3(17)
      abb3(21)=abb3(17)*abb3(5)
      abb3(22)=spal3l5*spbl3k2
      abb3(23)=abb3(22)*spak1k2
      abb3(24)=abb3(21)*abb3(23)
      abb3(25)=spbl3k2*abb3(4)
      abb3(26)=abb3(25)*spak1l3
      abb3(27)=abb3(26)*spbl4k2
      abb3(28)=-abb3(17)*abb3(27)
      abb3(24)=abb3(28)+abb3(24)-abb3(20)
      abb3(23)=abb3(5)*abb3(23)
      abb3(23)=-abb3(27)+abb3(23)
      abb3(23)=abb3(23)*abb3(17)
      abb3(20)=-abb3(20)+abb3(23)
      abb3(20)=2.0_ki*abb3(20)*abb3(6)**2
      abb3(23)=abb3(11)*abb3(16)
      abb3(27)=abb3(10)*abb3(2)
      abb3(28)=mT**2
      abb3(29)=abb3(27)*abb3(28)
      abb3(23)=abb3(23)+abb3(29)
      abb3(23)=abb3(23)*c2
      abb3(15)=abb3(15)*abb3(6)
      abb3(10)=abb3(10)*abb3(12)
      abb3(29)=abb3(10)*abb3(28)
      abb3(15)=abb3(15)+abb3(29)
      abb3(15)=abb3(15)*c1
      abb3(15)=abb3(23)-abb3(15)
      abb3(15)=abb3(6)*abb3(15)
      abb3(23)=spak1l5*abb3(15)
      abb3(29)=abb3(28)*abb3(11)
      abb3(30)=abb3(12)*c1
      abb3(31)=abb3(29)*abb3(30)
      abb3(32)=abb3(18)*c2
      abb3(33)=abb3(32)*abb3(28)
      abb3(31)=abb3(31)-abb3(33)
      abb3(26)=-abb3(31)*abb3(26)
      abb3(23)=abb3(23)+abb3(26)
      abb3(23)=8.0_ki*abb3(5)*abb3(23)
      abb3(26)=2.0_ki*abb3(5)
      abb3(33)=-abb3(15)*abb3(26)
      abb3(19)=abb3(33)+abb3(19)
      abb3(33)=2.0_ki*spak1k2
      abb3(19)=abb3(19)*abb3(33)
      abb3(33)=abb3(17)*spbl4k2
      abb3(26)=abb3(31)*abb3(26)
      abb3(26)=abb3(26)+abb3(33)
      abb3(25)=2.0_ki*abb3(25)
      abb3(26)=spak1k2*abb3(26)*abb3(25)
      abb3(11)=abb3(11)*abb3(30)
      abb3(11)=abb3(32)-abb3(11)
      abb3(22)=4.0_ki*abb3(11)*abb3(22)
      abb3(13)=-2.0_ki*abb3(13)
      abb3(17)=-abb3(17)*abb3(25)
      abb3(16)=c2*abb3(16)*abb3(9)
      abb3(14)=abb3(14)*c1
      abb3(25)=mT*abb3(14)
      abb3(16)=abb3(16)-abb3(25)
      abb3(16)=abb3(16)*abb3(5)
      abb3(25)=2.0_ki*spbl4k2
      abb3(11)=abb3(11)*abb3(25)
      abb3(11)=abb3(16)+abb3(11)
      abb3(16)=-2.0_ki*spal3l5*abb3(11)
      abb3(30)=4.0_ki*spal3l5*abb3(21)
      abb3(15)=-abb3(15)*abb3(25)
      abb3(25)=abb3(28)*abb3(6)
      abb3(31)=mT**3
      abb3(25)=abb3(25)+abb3(31)
      abb3(14)=abb3(25)*abb3(14)
      abb3(9)=-c2*abb3(25)*abb3(9)*abb3(2)
      abb3(9)=abb3(9)+abb3(14)
      abb3(9)=abb3(5)*abb3(9)
      abb3(9)=abb3(9)+abb3(15)
      abb3(9)=abb3(4)*abb3(9)
      abb3(14)=abb3(7)*abb3(8)*spak2l5*mH**2
      abb3(11)=-abb3(11)*abb3(14)
      abb3(9)=abb3(9)+abb3(11)
      abb3(9)=2.0_ki*abb3(9)
      abb3(11)=abb3(18)*abb3(28)
      abb3(15)=-abb3(31)*abb3(27)
      abb3(11)=abb3(15)-abb3(11)
      abb3(11)=c2*abb3(11)
      abb3(10)=abb3(31)*abb3(10)
      abb3(12)=abb3(12)*abb3(29)
      abb3(10)=abb3(10)+abb3(12)
      abb3(10)=c1*abb3(10)
      abb3(10)=abb3(11)+abb3(10)
      abb3(10)=abb3(4)*abb3(5)*abb3(10)
      abb3(11)=abb3(21)*abb3(14)
      abb3(10)=abb3(10)+abb3(11)
      abb3(10)=4.0_ki*abb3(10)
      R2d3=abb3(24)
      rat2 = rat2 + R2d3
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='3' value='", &
          & R2d3, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd3h5_qp
