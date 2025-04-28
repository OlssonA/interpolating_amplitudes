module     p2_gg_httbar_abbrevd81h8
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_kinematics, only: epstensor
   use p2_gg_httbar_globalsh8
   implicit none
   private
   complex(ki), dimension(58), public :: abb81
   complex(ki), public :: R2d81
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
      abb81(1)=1.0_ki/(-mT**2+es34)
      abb81(2)=sqrt(mT**2)
      abb81(3)=NC**(-1)
      abb81(4)=spak2l5**(-1)
      abb81(5)=spbl4k2**(-1)
      abb81(6)=spak2l3**(-1)
      abb81(7)=spbl3k2**(-1)
      abb81(8)=c2*abb81(3)
      abb81(8)=abb81(8)-c3
      abb81(9)=i_*TR*e*gHT*abb81(1)*gs**4
      abb81(10)=abb81(9)*abb81(2)
      abb81(11)=-abb81(10)*abb81(8)
      abb81(12)=mT**2
      abb81(13)=-abb81(12)*abb81(11)
      abb81(14)=spbl3k2*abb81(5)
      abb81(15)=abb81(14)*spae1l3
      abb81(16)=abb81(13)*abb81(15)
      abb81(17)=abb81(2)**2
      abb81(18)=abb81(17)*abb81(9)
      abb81(19)=abb81(10)*mT
      abb81(18)=abb81(19)+abb81(18)
      abb81(19)=-mT*abb81(8)
      abb81(18)=-abb81(18)*abb81(19)
      abb81(20)=abb81(18)*spae1l4
      abb81(16)=abb81(16)+abb81(20)
      abb81(20)=spbe2e1*abb81(4)
      abb81(21)=abb81(20)*spae2k2
      abb81(22)=abb81(21)*abb81(16)
      abb81(23)=abb81(18)*abb81(5)
      abb81(24)=abb81(23)*spbl5e1
      abb81(25)=spbk2e2*spae1e2
      abb81(26)=abb81(24)*abb81(25)
      abb81(27)=abb81(11)*spae1e2
      abb81(28)=abb81(27)*spbk2e2
      abb81(29)=abb81(28)*spbl5e1
      abb81(30)=spak2l4*mH**2*abb81(7)*abb81(6)
      abb81(31)=abb81(29)*abb81(30)
      abb81(22)=-abb81(22)+abb81(26)-abb81(31)
      abb81(26)=-es12*abb81(22)
      abb81(18)=abb81(18)*spae1k2
      abb81(31)=abb81(18)*spak1l4
      abb81(32)=-spbk2k1*abb81(31)
      abb81(33)=abb81(2)**3
      abb81(34)=abb81(33)*abb81(9)
      abb81(35)=abb81(8)*abb81(34)
      abb81(36)=abb81(35)*abb81(12)
      abb81(37)=-abb81(36)*abb81(15)
      abb81(32)=abb81(37)+abb81(32)
      abb81(32)=abb81(21)*abb81(32)
      abb81(37)=spbl5e1*abb81(5)*abb81(25)
      abb81(38)=-spae1l4*abb81(21)
      abb81(39)=spae2l4*spae1k2*abb81(20)
      abb81(37)=abb81(39)+abb81(37)+abb81(38)
      abb81(38)=abb81(9)*mT
      abb81(33)=abb81(33)*abb81(38)
      abb81(9)=abb81(9)*abb81(2)**4
      abb81(9)=abb81(9)+abb81(33)
      abb81(9)=-abb81(9)*abb81(19)
      abb81(19)=abb81(9)*abb81(37)
      abb81(33)=abb81(30)*spbl5e1
      abb81(37)=abb81(35)*abb81(25)*abb81(33)
      abb81(39)=spbk2e1*abb81(5)
      abb81(9)=-abb81(9)*abb81(39)
      abb81(40)=abb81(30)*spbk2e1
      abb81(41)=-abb81(35)*abb81(40)
      abb81(9)=abb81(9)+abb81(41)
      abb81(41)=spbl5e2*spae1e2
      abb81(9)=abb81(9)*abb81(41)
      abb81(42)=abb81(23)*spbk2e1
      abb81(25)=abb81(42)*abb81(25)
      abb81(43)=-abb81(28)*abb81(40)
      abb81(43)=abb81(25)+abb81(43)
      abb81(44)=spak1k2*spbl5k1
      abb81(43)=abb81(43)*abb81(44)
      abb81(35)=abb81(35)*spal3l4
      abb81(45)=-abb81(41)*abb81(35)
      abb81(44)=-spal3l4*abb81(28)*abb81(44)
      abb81(44)=abb81(45)+abb81(44)
      abb81(44)=spbl3e1*abb81(44)
      abb81(17)=abb81(38)*abb81(17)
      abb81(34)=abb81(34)+abb81(17)
      abb81(34)=-abb81(34)*abb81(8)
      abb81(45)=abb81(34)*abb81(41)
      abb81(46)=spbk1e1*spak1l4
      abb81(47)=abb81(46)*abb81(45)
      abb81(48)=abb81(34)*spae1l4
      abb81(17)=-abb81(17)*abb81(8)
      abb81(15)=abb81(17)*abb81(15)
      abb81(15)=abb81(15)+abb81(48)
      abb81(48)=spbe2e1*abb81(15)
      abb81(49)=spbl5k2*spae2k2
      abb81(50)=abb81(48)*abb81(49)
      abb81(34)=-abb81(12)*abb81(34)
      abb81(51)=abb81(5)*abb81(34)*abb81(21)
      abb81(52)=abb81(17)*abb81(21)
      abb81(53)=abb81(52)*abb81(30)
      abb81(51)=abb81(53)-abb81(51)
      abb81(53)=-spae1k1*spbk2k1*abb81(51)
      abb81(41)=abb81(14)*abb81(41)*abb81(17)
      abb81(54)=spbk1e1*abb81(41)
      abb81(55)=abb81(14)*spae1k2
      abb81(13)=abb81(55)*abb81(13)
      abb81(56)=abb81(13)*abb81(21)
      abb81(57)=-spbk2k1*abb81(56)
      abb81(54)=abb81(57)+abb81(54)
      abb81(54)=spak1l3*abb81(54)
      abb81(57)=spak1k2*abb81(29)
      abb81(58)=-spae1k1*abb81(52)
      abb81(57)=abb81(57)+abb81(58)
      abb81(57)=spbl3k1*spal3l4*abb81(57)
      abb81(36)=spae2l3*abb81(36)*abb81(20)*abb81(55)
      abb81(55)=spbl5e1*spae1e2
      abb81(35)=spbl3e2*abb81(55)*abb81(35)
      abb81(9)=abb81(36)+abb81(35)+abb81(57)+abb81(26)+abb81(54)+abb81(53)+abb8&
      &1(50)+abb81(47)+abb81(44)+abb81(43)+abb81(9)+abb81(37)+abb81(19)+abb81(3&
      &2)
      abb81(19)=2.0_ki*abb81(22)
      abb81(26)=spae2l4*abb81(18)
      abb81(32)=spae2l3*abb81(13)
      abb81(26)=abb81(32)+abb81(26)
      abb81(26)=abb81(20)*abb81(26)
      abb81(32)=spbl3e1*spal3l4
      abb81(35)=spbl5e2*abb81(32)
      abb81(36)=spal3l4*spbl5e1
      abb81(37)=-spbl3e2*abb81(36)
      abb81(35)=abb81(37)+abb81(35)
      abb81(35)=abb81(27)*abb81(35)
      abb81(37)=abb81(40)*abb81(27)
      abb81(43)=abb81(42)*spae1e2
      abb81(37)=abb81(37)-abb81(43)
      abb81(43)=spbl5e2*abb81(37)
      abb81(22)=abb81(43)+abb81(22)+abb81(35)+abb81(26)
      abb81(26)=abb81(13)*spak1l3
      abb81(26)=abb81(26)+abb81(31)
      abb81(31)=abb81(4)*abb81(26)
      abb81(35)=abb81(16)*abb81(4)
      abb81(43)=-spak1k2*abb81(35)
      abb81(31)=abb81(43)+abb81(31)
      abb81(31)=spbe2k1*abb81(31)
      abb81(15)=-spbl5e2*abb81(15)
      abb81(15)=abb81(31)+abb81(15)
      abb81(31)=abb81(40)*abb81(11)
      abb81(43)=abb81(11)*spal3l4
      abb81(44)=abb81(43)*spbl3e1
      abb81(31)=abb81(44)+abb81(31)-abb81(42)
      abb81(42)=spbl5k1*abb81(31)
      abb81(11)=abb81(11)*abb81(33)
      abb81(11)=abb81(11)-abb81(24)
      abb81(24)=-spbk2k1*abb81(11)
      abb81(43)=abb81(43)*spbl5e1
      abb81(47)=-spbl3k1*abb81(43)
      abb81(24)=abb81(47)+abb81(24)+abb81(42)
      abb81(24)=spak1e2*abb81(24)
      abb81(40)=abb81(32)+abb81(40)
      abb81(17)=abb81(40)*abb81(17)
      abb81(34)=abb81(34)*abb81(39)
      abb81(17)=abb81(17)-abb81(34)
      abb81(34)=spae2k2*abb81(4)
      abb81(39)=abb81(34)*abb81(17)
      abb81(10)=abb81(38)+abb81(10)
      abb81(10)=-abb81(10)*abb81(8)
      abb81(42)=abb81(46)*abb81(10)
      abb81(46)=abb81(49)*abb81(42)
      abb81(8)=-abb81(38)*abb81(8)
      abb81(14)=abb81(8)*abb81(14)
      abb81(38)=abb81(49)*abb81(14)
      abb81(47)=spak1l3*spbk1e1
      abb81(50)=abb81(38)*abb81(47)
      abb81(24)=abb81(50)+abb81(46)+abb81(39)+abb81(24)
      abb81(39)=spae1k2*spbk2e2
      abb81(44)=-abb81(39)*abb81(44)
      abb81(44)=abb81(44)-abb81(48)
      abb81(46)=-abb81(14)*abb81(47)
      abb81(31)=abb81(46)-2.0_ki*abb81(31)-abb81(42)
      abb81(42)=2.0_ki*abb81(4)
      abb81(46)=-abb81(18)*abb81(42)
      abb81(47)=abb81(10)*abb81(49)
      abb81(48)=abb81(39)*abb81(43)
      abb81(43)=2.0_ki*abb81(43)
      abb81(13)=-abb81(13)*abb81(42)
      abb81(42)=-spal3l4*abb81(52)
      abb81(39)=abb81(39)*abb81(11)
      abb81(11)=2.0_ki*abb81(11)
      abb81(17)=spae1e2*abb81(4)*abb81(17)
      abb81(34)=abb81(34)*spbk2e1*abb81(16)
      abb81(17)=abb81(17)+abb81(34)
      abb81(12)=abb81(4)*abb81(5)*abb81(10)*abb81(12)
      abb81(8)=abb81(8)*abb81(4)
      abb81(30)=abb81(8)*abb81(30)
      abb81(12)=abb81(12)+abb81(30)
      abb81(30)=spbk2k1*abb81(12)
      abb81(8)=abb81(8)*spal3l4
      abb81(34)=spbl3k1*abb81(8)
      abb81(30)=abb81(34)+abb81(30)
      abb81(30)=spae1k1*abb81(30)
      abb81(30)=2.0_ki*abb81(35)+abb81(30)
      abb81(16)=spak1k2*abb81(16)
      abb81(16)=abb81(16)-abb81(26)
      abb81(16)=abb81(20)*abb81(16)
      abb81(20)=-spbl5k1*abb81(32)
      abb81(26)=spbl3k1*abb81(36)
      abb81(20)=abb81(26)+abb81(20)
      abb81(20)=abb81(27)*abb81(20)
      abb81(23)=-abb81(55)*abb81(23)
      abb81(26)=abb81(27)*abb81(33)
      abb81(23)=abb81(23)+abb81(26)
      abb81(23)=spbk2k1*abb81(23)
      abb81(26)=-spbl5k1*abb81(37)
      abb81(20)=abb81(23)+abb81(26)+abb81(20)
      abb81(18)=abb81(21)*abb81(18)
      abb81(21)=abb81(28)*abb81(40)
      abb81(21)=-abb81(25)+abb81(21)
      abb81(23)=-spal3l4*abb81(29)
      R2d81=0.0_ki
      rat2 = rat2 + R2d81
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='81' value='", &
          & R2d81, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd81h8
