module     p0_ubaru_httbar_abbrevd72h2_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh2_qp
   implicit none
   private
   complex(ki), dimension(51), public :: abb72
   complex(ki), public :: R2d72
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
      abb72(1)=1.0_ki/(-mT**2+es34)
      abb72(2)=NC**(-1)
      abb72(3)=es12**(-1)
      abb72(4)=spak2l3**(-1)
      abb72(5)=spbl3k2**(-1)
      abb72(6)=spbl4k2**(-1)
      abb72(7)=spbl5k2**(-1)
      abb72(8)=sqrt(mT**2)
      abb72(9)=spak2l4**(-1)
      abb72(10)=i_*e*gHT*abb72(3)*TR**2*gs**4
      abb72(11)=abb72(10)*spbk2k1*abb72(1)
      abb72(12)=abb72(11)*spak2l5
      abb72(13)=abb72(8)*mT
      abb72(14)=abb72(12)*abb72(13)
      abb72(15)=mT**2
      abb72(16)=abb72(12)*abb72(15)
      abb72(17)=abb72(14)+abb72(16)
      abb72(18)=abb72(6)*c2
      abb72(19)=abb72(18)*abb72(17)
      abb72(20)=abb72(11)*abb72(13)
      abb72(15)=abb72(15)*abb72(11)
      abb72(21)=abb72(20)+abb72(15)
      abb72(22)=spak2l4*abb72(7)
      abb72(23)=abb72(22)*c2
      abb72(24)=abb72(23)*abb72(21)
      abb72(19)=abb72(19)+abb72(24)
      abb72(19)=abb72(19)*NC
      abb72(25)=abb72(2)**2
      abb72(25)=abb72(25)-1.0_ki
      abb72(26)=abb72(25)*c1
      abb72(27)=c2*abb72(2)
      abb72(28)=abb72(26)-abb72(27)
      abb72(29)=abb72(15)*abb72(28)
      abb72(30)=abb72(7)*abb72(6)
      abb72(31)=abb72(29)*abb72(30)
      abb72(32)=abb72(30)*abb72(15)
      abb72(33)=NC*c2
      abb72(34)=abb72(32)*abb72(33)
      abb72(34)=abb72(31)+abb72(34)
      abb72(35)=spak2l3*spbl3k2
      abb72(34)=abb72(34)*abb72(35)
      abb72(36)=abb72(2)-NC
      abb72(10)=abb72(10)*spak2l5*abb72(1)
      abb72(36)=abb72(36)*c2*abb72(10)
      abb72(37)=c1*abb72(25)*abb72(10)
      abb72(36)=abb72(36)-abb72(37)
      abb72(37)=spbl3k1*spal3l4
      abb72(38)=-abb72(36)*abb72(37)
      abb72(34)=abb72(34)+abb72(38)
      abb72(38)=-abb72(6)*abb72(17)*abb72(28)
      abb72(25)=abb72(12)*abb72(25)
      abb72(39)=abb72(25)*c1
      abb72(27)=abb72(27)*abb72(12)
      abb72(39)=abb72(39)-abb72(27)
      abb72(40)=mH**2
      abb72(41)=abb72(40)*abb72(5)
      abb72(42)=abb72(41)*spak2l4
      abb72(43)=abb72(39)*abb72(42)
      abb72(12)=abb72(33)*abb72(12)
      abb72(44)=abb72(12)*abb72(42)
      abb72(43)=abb72(43)+abb72(44)
      abb72(43)=abb72(43)*abb72(4)
      abb72(45)=abb72(22)*abb72(21)*abb72(28)
      abb72(19)=abb72(34)+abb72(45)-abb72(38)+abb72(19)+abb72(43)
      abb72(38)=2.0_ki*abb72(19)
      abb72(43)=abb72(8)**2
      abb72(46)=-abb72(42)*abb72(43)*abb72(39)
      abb72(47)=abb72(20)*abb72(28)
      abb72(48)=abb72(33)*abb72(20)
      abb72(49)=abb72(47)+abb72(48)
      abb72(49)=abb72(7)*abb72(49)
      abb72(50)=spbl4k2*abb72(41)*abb72(49)*spak2l4**2
      abb72(44)=-abb72(43)*abb72(44)
      abb72(44)=abb72(50)+abb72(46)+abb72(44)
      abb72(44)=abb72(4)*abb72(44)
      abb72(46)=abb72(47)*abb72(22)
      abb72(47)=abb72(48)*abb72(22)
      abb72(47)=abb72(47)+abb72(46)
      abb72(48)=spbl4l3*abb72(47)
      abb72(10)=abb72(13)*abb72(10)
      abb72(50)=abb72(6)*abb72(10)*abb72(28)
      abb72(33)=abb72(33)*abb72(6)
      abb72(10)=abb72(33)*abb72(10)
      abb72(10)=abb72(50)+abb72(10)
      abb72(50)=abb72(10)*spbl4k1
      abb72(51)=-spbl3k2*abb72(50)
      abb72(48)=abb72(51)+abb72(48)
      abb72(48)=spal3l4*abb72(48)
      abb72(27)=-abb72(12)+abb72(27)
      abb72(51)=abb72(8)*mT**3
      abb72(27)=abb72(51)*abb72(27)
      abb72(25)=-c1*abb72(51)*abb72(25)
      abb72(25)=abb72(25)+abb72(27)
      abb72(25)=abb72(9)*abb72(25)*abb72(6)**2
      abb72(11)=abb72(11)*abb72(13)**2
      abb72(13)=abb72(26)*abb72(11)
      abb72(11)=c2*abb72(11)
      abb72(26)=abb72(11)*abb72(2)
      abb72(13)=abb72(13)-abb72(26)
      abb72(26)=-NC*abb72(11)
      abb72(26)=-abb72(13)+abb72(26)
      abb72(26)=abb72(30)*abb72(26)
      abb72(26)=-abb72(25)+abb72(26)
      abb72(26)=abb72(35)*abb72(26)
      abb72(21)=-abb72(22)*abb72(28)*abb72(21)*abb72(43)
      abb72(17)=abb72(17)*abb72(43)
      abb72(27)=-abb72(6)*abb72(17)*abb72(28)
      abb72(21)=abb72(27)+abb72(21)
      abb72(27)=abb72(6)*abb72(14)*abb72(28)
      abb72(27)=abb72(27)+abb72(46)
      abb72(27)=abb72(27)*abb72(40)
      abb72(17)=-abb72(18)*abb72(17)
      abb72(35)=-abb72(43)*abb72(24)
      abb72(17)=abb72(17)+abb72(35)
      abb72(35)=abb72(14)*abb72(18)
      abb72(20)=abb72(20)*abb72(23)
      abb72(20)=abb72(35)+abb72(20)
      abb72(20)=abb72(20)*abb72(40)
      abb72(17)=2.0_ki*abb72(17)+abb72(20)
      abb72(17)=NC*abb72(17)
      abb72(20)=abb72(43)*abb72(36)
      abb72(23)=abb72(20)*abb72(37)
      abb72(17)=abb72(23)+abb72(17)+2.0_ki*abb72(21)+abb72(27)+abb72(26)+abb72(&
      &48)+abb72(44)
      abb72(17)=2.0_ki*abb72(17)
      abb72(21)=8.0_ki*abb72(47)
      abb72(23)=-4.0_ki*abb72(34)
      abb72(14)=abb72(14)-abb72(16)
      abb72(18)=abb72(14)*abb72(18)
      abb72(18)=abb72(18)-abb72(24)
      abb72(18)=NC*abb72(18)
      abb72(14)=abb72(6)*abb72(14)*abb72(28)
      abb72(14)=abb72(18)+abb72(14)-abb72(45)
      abb72(14)=abb72(41)*abb72(14)
      abb72(12)=abb72(39)+abb72(12)
      abb72(18)=-abb72(4)*spak2l4*abb72(12)*abb72(5)**2*mH**4
      abb72(14)=abb72(18)+abb72(14)
      abb72(14)=abb72(4)*abb72(14)
      abb72(18)=2.0_ki*abb72(30)
      abb72(13)=abb72(13)*abb72(18)
      abb72(24)=-abb72(40)*abb72(31)
      abb72(11)=abb72(11)*abb72(18)
      abb72(18)=-c2*abb72(40)*abb72(32)
      abb72(11)=abb72(11)+abb72(18)
      abb72(11)=NC*abb72(11)
      abb72(11)=-2.0_ki*abb72(25)+abb72(14)+abb72(11)+abb72(13)+abb72(24)
      abb72(11)=4.0_ki*abb72(11)
      abb72(13)=-4.0_ki*abb72(19)
      abb72(14)=-abb72(6)*abb72(16)*abb72(28)
      abb72(16)=abb72(33)*abb72(16)
      abb72(14)=abb72(14)-abb72(16)
      abb72(14)=abb72(14)*abb72(9)
      abb72(16)=abb72(14)+abb72(49)
      abb72(16)=spak2l3*abb72(16)
      abb72(12)=abb72(12)*abb72(41)
      abb72(16)=-abb72(12)+abb72(16)
      abb72(16)=2.0_ki*abb72(16)
      abb72(18)=-spbl3k1*abb72(41)*abb72(36)
      abb72(18)=abb72(18)+2.0_ki*abb72(50)
      abb72(18)=2.0_ki*abb72(18)
      abb72(19)=4.0_ki*abb72(20)
      abb72(20)=abb72(29)*abb72(6)
      abb72(15)=abb72(33)*abb72(15)
      abb72(15)=abb72(15)+abb72(20)
      abb72(15)=abb72(15)*abb72(22)
      abb72(20)=2.0_ki*spbl3k2
      abb72(22)=-abb72(15)*abb72(20)
      abb72(24)=abb72(4)*abb72(42)*abb72(36)
      abb72(24)=abb72(24)+abb72(10)
      abb72(24)=spbl3k1*abb72(24)
      abb72(15)=-spbl4l3*abb72(15)
      abb72(15)=abb72(24)+abb72(15)
      abb72(15)=2.0_ki*abb72(15)
      abb72(10)=abb72(10)*abb72(20)
      abb72(12)=-abb72(4)*abb72(12)
      abb72(12)=abb72(14)+abb72(12)-abb72(49)
      abb72(12)=2.0_ki*spal3l4*abb72(12)
      R2d72=abb72(38)
      rat2 = rat2 + R2d72
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='72' value='", &
          & R2d72, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd72h2_qp
