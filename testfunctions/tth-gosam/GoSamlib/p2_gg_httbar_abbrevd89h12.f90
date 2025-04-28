module     p2_gg_httbar_abbrevd89h12
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_kinematics, only: epstensor
   use p2_gg_httbar_globalsh12
   implicit none
   private
   complex(ki), dimension(56), public :: abb89
   complex(ki), public :: R2d89
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p2_gg_httbar_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_model
      use p2_gg_httbar_color, only: TR
      use p2_gg_httbar_globalsl1, only: epspow
      implicit none
      abb89(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb89(2)=sqrt(mT**2)
      abb89(3)=NC**(-1)
      abb89(4)=spak2l4**(-1)
      abb89(5)=spak2l5**(-1)
      abb89(6)=spak2l3**(-1)
      abb89(7)=spbl3k2**(-1)
      abb89(8)=mT*abb89(2)
      abb89(9)=i_*TR*e*gHT*abb89(1)*gs**4
      abb89(10)=abb89(8)*abb89(9)
      abb89(11)=abb89(9)*abb89(2)**2
      abb89(12)=abb89(11)+abb89(10)
      abb89(13)=c2*abb89(3)
      abb89(13)=abb89(13)-c3
      abb89(12)=-abb89(12)*abb89(13)
      abb89(14)=abb89(12)*spbk2e2
      abb89(15)=spbl5k1*abb89(14)
      abb89(10)=-abb89(10)*abb89(13)
      abb89(16)=spak2l3*abb89(5)
      abb89(17)=abb89(10)*abb89(16)
      abb89(18)=abb89(17)*spbk2e2
      abb89(19)=spbl3k1*abb89(18)
      abb89(15)=abb89(19)+abb89(15)
      abb89(15)=spak1k2*abb89(15)
      abb89(19)=-abb89(11)*abb89(13)
      abb89(20)=abb89(19)*spbl5l3
      abb89(21)=spbk2e2*spak2l3
      abb89(22)=abb89(20)*abb89(21)
      abb89(23)=abb89(2)**3
      abb89(24)=abb89(9)*mT
      abb89(25)=abb89(23)*abb89(24)
      abb89(26)=abb89(9)*abb89(2)**4
      abb89(25)=abb89(25)+abb89(26)
      abb89(25)=abb89(13)*abb89(25)
      abb89(26)=spbl5e2*abb89(25)
      abb89(15)=abb89(26)+abb89(22)+abb89(15)
      abb89(22)=spbl4e1*spae1e2
      abb89(15)=abb89(22)*abb89(15)
      abb89(26)=abb89(17)*spbl3e1
      abb89(27)=abb89(12)*spbl5e1
      abb89(26)=abb89(26)+abb89(27)
      abb89(27)=spae1e2*abb89(26)
      abb89(28)=abb89(27)*spbk2e2
      abb89(29)=-spak1k2*spbl4k1*abb89(28)
      abb89(30)=abb89(23)*abb89(9)
      abb89(11)=abb89(11)*mT
      abb89(11)=abb89(11)+abb89(30)
      abb89(30)=-mT*abb89(13)
      abb89(11)=-abb89(11)*abb89(30)
      abb89(31)=abb89(11)*abb89(5)
      abb89(32)=abb89(7)*mH**2*abb89(6)*spbl5k2
      abb89(19)=abb89(32)*abb89(19)
      abb89(19)=abb89(31)-abb89(19)
      abb89(31)=abb89(19)*spae2k2
      abb89(33)=spbe2e1*abb89(31)
      abb89(34)=spae1k1*spbl4k1
      abb89(35)=-abb89(34)*abb89(33)
      abb89(25)=-spbl5e1*abb89(25)
      abb89(13)=-abb89(9)*abb89(13)
      abb89(23)=-abb89(13)*abb89(23)*mT
      abb89(36)=abb89(23)*abb89(16)
      abb89(37)=-spbl3e1*abb89(36)
      abb89(25)=abb89(25)+abb89(37)
      abb89(25)=spbl4e2*spae1e2*abb89(25)
      abb89(8)=-abb89(13)*abb89(8)**2
      abb89(37)=abb89(8)*spae1e2
      abb89(38)=spbk1e1*spak1k2
      abb89(39)=abb89(16)*abb89(4)
      abb89(40)=abb89(38)*abb89(39)
      abb89(41)=-abb89(40)*abb89(37)
      abb89(36)=abb89(22)*abb89(36)
      abb89(36)=abb89(36)+abb89(41)
      abb89(36)=spbl3e2*abb89(36)
      abb89(41)=spbl5l3*abb89(4)
      abb89(42)=abb89(41)*spbe2e1
      abb89(23)=abb89(23)*abb89(42)
      abb89(43)=-spae1k2*abb89(23)
      abb89(44)=abb89(34)*abb89(20)*spbe2e1
      abb89(43)=abb89(43)+abb89(44)
      abb89(43)=spae2l3*abb89(43)
      abb89(44)=spae1k2*abb89(4)
      abb89(45)=spbe2e1*abb89(44)
      abb89(46)=abb89(45)*abb89(11)
      abb89(47)=spbl5k2*spae2k2
      abb89(48)=-abb89(47)*abb89(46)
      abb89(11)=abb89(11)*spbl5e2
      abb89(49)=abb89(4)*abb89(11)*spae1e2
      abb89(50)=-abb89(38)*abb89(49)
      abb89(51)=spae1l3*spae2k2
      abb89(23)=abb89(51)*abb89(23)
      abb89(8)=abb89(8)*abb89(16)
      abb89(16)=abb89(8)*abb89(45)
      abb89(45)=spbl3k2*spae2k2
      abb89(52)=-abb89(45)*abb89(16)
      abb89(42)=abb89(42)*abb89(10)
      abb89(53)=abb89(42)*spae2k2
      abb89(54)=abb89(53)*spae1k2
      abb89(55)=-spbk2k1*spak1l3*abb89(54)
      abb89(53)=abb89(53)*spae1l3
      abb89(56)=es12*abb89(53)
      abb89(15)=abb89(56)+abb89(55)+abb89(52)+abb89(43)+abb89(36)+abb89(25)+abb&
      &89(23)+abb89(50)+abb89(48)+abb89(15)+abb89(35)+abb89(29)
      abb89(23)=2.0_ki*abb89(53)
      abb89(25)=-spbl5e2*abb89(12)
      abb89(29)=-spbl3e2*abb89(17)
      abb89(25)=abb89(29)+abb89(25)
      abb89(25)=abb89(22)*abb89(25)
      abb89(29)=spbl4e2*abb89(27)
      abb89(35)=spae2l3*spae1k2*abb89(42)
      abb89(25)=abb89(35)+abb89(29)-abb89(53)+abb89(25)
      abb89(8)=-spbl3e2*abb89(8)
      abb89(8)=abb89(8)-abb89(11)
      abb89(8)=abb89(8)*abb89(44)
      abb89(10)=abb89(10)*abb89(41)
      abb89(11)=abb89(10)*spae1l3
      abb89(29)=spak1k2*abb89(11)
      abb89(35)=spae1k2*abb89(10)
      abb89(36)=-spak1l3*abb89(35)
      abb89(29)=abb89(29)+abb89(36)
      abb89(29)=spbe2k1*abb89(29)
      abb89(36)=abb89(13)*spbl5l3
      abb89(21)=abb89(21)*abb89(36)
      abb89(41)=-abb89(34)*abb89(21)
      abb89(8)=abb89(29)+abb89(41)+abb89(8)
      abb89(29)=abb89(17)*spbl3k1
      abb89(41)=abb89(12)*spbl5k1
      abb89(29)=abb89(29)+abb89(41)
      abb89(41)=spak1e2*abb89(29)
      abb89(43)=abb89(20)*spae2l3
      abb89(31)=abb89(43)+abb89(41)-abb89(31)
      abb89(31)=spbl4e1*abb89(31)
      abb89(41)=-spak1e2*spbl4k1*abb89(26)
      abb89(44)=-abb89(13)*mT**2
      abb89(45)=abb89(45)*abb89(44)
      abb89(48)=abb89(40)*abb89(45)
      abb89(9)=abb89(9)*abb89(2)
      abb89(9)=abb89(9)+abb89(24)
      abb89(9)=-abb89(9)*abb89(30)
      abb89(24)=abb89(9)*abb89(4)
      abb89(30)=abb89(47)*abb89(24)
      abb89(47)=abb89(38)*abb89(30)
      abb89(31)=abb89(48)+abb89(41)+abb89(47)+abb89(31)
      abb89(41)=spae1k2*spbl4e1
      abb89(47)=-abb89(14)*abb89(41)
      abb89(46)=-abb89(46)+abb89(47)
      abb89(47)=2.0_ki*spbl4e1
      abb89(12)=abb89(12)*abb89(47)
      abb89(38)=abb89(38)*abb89(24)
      abb89(12)=abb89(12)+abb89(38)
      abb89(38)=spbk2e2*spae1k2*abb89(26)
      abb89(26)=-2.0_ki*abb89(26)
      abb89(43)=-spbe2e1*abb89(43)
      abb89(33)=abb89(43)+abb89(33)
      abb89(41)=-abb89(18)*abb89(41)
      abb89(16)=-abb89(16)+abb89(41)
      abb89(40)=abb89(44)*abb89(40)
      abb89(17)=abb89(17)*abb89(47)
      abb89(17)=abb89(17)+abb89(40)
      abb89(20)=abb89(22)*abb89(20)
      abb89(40)=-abb89(34)*abb89(36)
      abb89(35)=-2.0_ki*abb89(35)+abb89(40)
      abb89(19)=-abb89(22)*abb89(19)
      abb89(10)=-spbk2e1*abb89(51)*abb89(10)
      abb89(10)=abb89(10)+abb89(19)
      abb89(9)=abb89(9)*abb89(5)
      abb89(13)=abb89(13)*abb89(32)
      abb89(9)=abb89(9)-abb89(13)
      abb89(13)=abb89(34)*abb89(9)
      abb89(11)=2.0_ki*abb89(11)+abb89(13)
      abb89(13)=spbl3e2*abb89(39)*abb89(37)
      abb89(13)=abb89(49)+abb89(13)
      abb89(19)=-abb89(39)*abb89(45)
      abb89(19)=-abb89(30)+abb89(19)
      abb89(30)=-abb89(44)*abb89(39)
      abb89(32)=spak1l3*spae1k2
      abb89(34)=-spae1l3*spak1k2
      abb89(32)=abb89(34)+abb89(32)
      abb89(32)=abb89(42)*abb89(32)
      abb89(27)=spbl4k1*abb89(27)
      abb89(29)=-abb89(22)*abb89(29)
      abb89(27)=abb89(29)+abb89(27)
      abb89(14)=abb89(22)*abb89(14)
      abb89(18)=abb89(22)*abb89(18)
      R2d89=0.0_ki
      rat2 = rat2 + R2d89
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='89' value='", &
          & R2d89, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd89h12
