module     p2_gg_httbar_abbrevd89h8
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_kinematics, only: epstensor
   use p2_gg_httbar_globalsh8
   implicit none
   private
   complex(ki), dimension(57), public :: abb89
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
      abb89(4)=spak2l3**(-1)
      abb89(5)=spbl3k2**(-1)
      abb89(6)=spak2l5**(-1)
      abb89(7)=spbl4k2**(-1)
      abb89(8)=abb89(2)**2
      abb89(9)=i_*TR*e*gHT*abb89(1)*gs**4
      abb89(10)=abb89(8)*abb89(9)
      abb89(11)=abb89(9)*abb89(2)
      abb89(12)=abb89(11)*mT
      abb89(10)=abb89(12)+abb89(10)
      abb89(12)=c2*abb89(3)
      abb89(12)=abb89(12)-c3
      abb89(13)=-mT*abb89(12)
      abb89(10)=-abb89(10)*abb89(13)
      abb89(14)=abb89(10)*spbk2e1
      abb89(15)=abb89(14)*spbl5k1
      abb89(16)=-abb89(11)*abb89(12)
      abb89(17)=mT**2
      abb89(18)=-abb89(17)*abb89(16)
      abb89(19)=spak2l3*abb89(6)
      abb89(20)=abb89(19)*spbk2e1
      abb89(21)=abb89(20)*abb89(18)
      abb89(22)=abb89(21)*spbl3k1
      abb89(15)=abb89(15)+abb89(22)
      abb89(23)=spak1k2*abb89(15)
      abb89(24)=abb89(9)*mT
      abb89(8)=abb89(24)*abb89(8)
      abb89(25)=-abb89(8)*abb89(12)
      abb89(26)=spbl5l3*spbk2e1
      abb89(27)=-spak2l3*abb89(25)*abb89(26)
      abb89(28)=abb89(2)**3
      abb89(29)=abb89(28)*abb89(24)
      abb89(30)=abb89(9)*abb89(2)**4
      abb89(29)=abb89(30)+abb89(29)
      abb89(13)=-abb89(29)*abb89(13)
      abb89(29)=spbl5e1*abb89(13)
      abb89(9)=abb89(28)*abb89(9)
      abb89(28)=abb89(12)*abb89(9)
      abb89(30)=abb89(28)*abb89(17)
      abb89(31)=abb89(19)*spbl3e1
      abb89(32)=abb89(30)*abb89(31)
      abb89(23)=abb89(32)+abb89(29)+abb89(27)+abb89(23)
      abb89(27)=spae1e2*abb89(7)
      abb89(29)=abb89(27)*spbk2e2
      abb89(23)=abb89(29)*abb89(23)
      abb89(32)=spbl5k2*mH**2*abb89(5)*abb89(4)
      abb89(33)=abb89(28)*abb89(32)
      abb89(34)=abb89(13)*abb89(6)
      abb89(33)=abb89(33)+abb89(34)
      abb89(8)=abb89(9)+abb89(8)
      abb89(8)=-abb89(8)*abb89(12)
      abb89(9)=spbl5k2*abb89(8)
      abb89(34)=abb89(25)*abb89(19)
      abb89(35)=spbl3k2*abb89(34)
      abb89(36)=abb89(16)*spbl5l3
      abb89(37)=abb89(36)*spak1l3
      abb89(38)=-spbk2k1*abb89(37)
      abb89(9)=abb89(38)+abb89(35)+abb89(9)-abb89(33)
      abb89(35)=spae2k2*spbe2e1
      abb89(38)=abb89(35)*spae1l4
      abb89(9)=abb89(38)*abb89(9)
      abb89(39)=abb89(32)*abb89(16)
      abb89(40)=abb89(10)*abb89(6)
      abb89(39)=abb89(39)-abb89(40)
      abb89(41)=abb89(38)*abb89(39)
      abb89(42)=abb89(10)*spbl5e1
      abb89(18)=abb89(31)*abb89(18)
      abb89(18)=abb89(42)+abb89(18)
      abb89(31)=abb89(29)*abb89(18)
      abb89(31)=abb89(31)+abb89(41)
      abb89(41)=-es12*abb89(31)
      abb89(42)=-spae1k2*abb89(39)
      abb89(43)=abb89(35)*spbk2k1*spak1l4
      abb89(44)=-abb89(42)*abb89(43)
      abb89(45)=abb89(8)*spbl5e2
      abb89(46)=abb89(45)*spae1e2
      abb89(47)=spbk1e1*spak1l4
      abb89(48)=abb89(47)*abb89(46)
      abb89(17)=abb89(17)*abb89(6)
      abb89(49)=-abb89(8)*abb89(17)
      abb89(50)=abb89(49)*abb89(7)
      abb89(51)=abb89(32)*abb89(7)
      abb89(52)=abb89(51)*abb89(25)
      abb89(50)=abb89(52)-abb89(50)
      abb89(50)=abb89(35)*abb89(50)
      abb89(52)=spae1k1*spbk2k1
      abb89(53)=-abb89(50)*abb89(52)
      abb89(54)=spae2l4*spbe2e1
      abb89(33)=abb89(54)*spae1k2*abb89(33)
      abb89(43)=abb89(36)*abb89(43)
      abb89(28)=abb89(28)*spbl5l3
      abb89(54)=abb89(28)*abb89(54)
      abb89(43)=abb89(43)+abb89(54)
      abb89(43)=spae1l3*abb89(43)
      abb89(20)=-abb89(30)*abb89(27)*abb89(20)
      abb89(30)=abb89(47)*abb89(34)*spae1e2
      abb89(20)=abb89(20)+abb89(30)
      abb89(20)=spbl3e2*abb89(20)
      abb89(30)=spae1l4*spbe2e1
      abb89(28)=-abb89(30)*abb89(28)
      abb89(54)=spbl5l3*abb89(7)
      abb89(55)=abb89(54)*abb89(25)
      abb89(56)=-abb89(52)*abb89(55)*spbe2e1
      abb89(28)=abb89(28)+abb89(56)
      abb89(28)=spae2l3*abb89(28)
      abb89(13)=-spbl5e2*spbk2e1*abb89(13)*abb89(27)
      abb89(9)=abb89(28)+abb89(20)+abb89(41)+abb89(43)+abb89(33)+abb89(53)+abb8&
      &9(48)+abb89(13)+abb89(44)+abb89(9)+abb89(23)
      abb89(13)=-2.0_ki*abb89(31)
      abb89(20)=abb89(36)*spae1l3
      abb89(23)=abb89(42)-abb89(20)
      abb89(28)=abb89(23)*spbe2e1
      abb89(33)=spae2l4*abb89(28)
      abb89(41)=-spbl5e2*abb89(14)
      abb89(42)=-spbl3e2*abb89(21)
      abb89(41)=abb89(42)+abb89(41)
      abb89(41)=abb89(27)*abb89(41)
      abb89(42)=spae2l3*abb89(30)*abb89(36)
      abb89(31)=abb89(42)+abb89(41)+abb89(33)+abb89(31)
      abb89(33)=abb89(34)*spbl3e2
      abb89(41)=-spbe2k1*abb89(37)
      abb89(41)=abb89(41)+abb89(33)+abb89(45)
      abb89(41)=spae1l4*abb89(41)
      abb89(42)=-spak1l4*abb89(23)
      abb89(43)=abb89(39)*spae1l4
      abb89(44)=-spak1k2*abb89(43)
      abb89(42)=abb89(44)+abb89(42)
      abb89(42)=spbe2k1*abb89(42)
      abb89(44)=-abb89(24)*abb89(12)
      abb89(45)=abb89(54)*abb89(44)
      abb89(48)=abb89(45)*abb89(52)
      abb89(53)=spbk2e2*spak2l3
      abb89(54)=abb89(48)*abb89(53)
      abb89(41)=abb89(42)+abb89(54)+abb89(41)
      abb89(42)=abb89(7)*abb89(18)
      abb89(54)=-spbk2k1*abb89(42)
      abb89(56)=spbk2e1*abb89(7)
      abb89(10)=abb89(56)*abb89(10)
      abb89(57)=spbl5k1*abb89(10)
      abb89(22)=abb89(7)*abb89(22)
      abb89(22)=abb89(22)+abb89(57)+abb89(54)
      abb89(22)=spak1e2*abb89(22)
      abb89(54)=-abb89(25)*abb89(32)
      abb89(54)=abb89(49)+abb89(54)
      abb89(54)=abb89(56)*abb89(54)
      abb89(19)=abb89(44)*abb89(19)
      abb89(56)=abb89(19)*spbl3k2
      abb89(11)=abb89(24)+abb89(11)
      abb89(11)=-abb89(11)*abb89(12)
      abb89(12)=abb89(11)*spbl5k2
      abb89(12)=abb89(56)+abb89(12)
      abb89(24)=-abb89(47)*abb89(12)
      abb89(24)=abb89(24)+abb89(54)
      abb89(24)=spae2k2*abb89(24)
      abb89(54)=abb89(55)*spae2l3
      abb89(55)=-spbk2e1*abb89(54)
      abb89(22)=abb89(55)+abb89(24)+abb89(22)
      abb89(8)=abb89(8)*abb89(30)
      abb89(24)=-abb89(11)*abb89(47)
      abb89(10)=2.0_ki*abb89(10)+abb89(24)
      abb89(20)=-spbk2e1*spae2k2*abb89(20)
      abb89(24)=-2.0_ki*abb89(23)
      abb89(33)=-spae1e2*abb89(33)
      abb89(33)=-abb89(46)+abb89(33)
      abb89(12)=spae2k2*abb89(12)
      abb89(34)=abb89(30)*abb89(34)
      abb89(46)=abb89(7)*abb89(21)
      abb89(47)=-abb89(47)*abb89(19)
      abb89(46)=2.0_ki*abb89(46)+abb89(47)
      abb89(47)=spae1l4*spae2k2
      abb89(16)=abb89(47)*abb89(16)
      abb89(25)=abb89(25)*abb89(27)
      abb89(16)=abb89(16)-abb89(25)
      abb89(25)=abb89(16)*abb89(26)
      abb89(26)=spae1l4*abb89(36)
      abb89(26)=-2.0_ki*abb89(26)+abb89(48)
      abb89(48)=spae1k2*spbk2e2*abb89(42)
      abb89(42)=-2.0_ki*abb89(42)
      abb89(49)=abb89(27)*abb89(49)
      abb89(16)=abb89(16)*abb89(32)
      abb89(32)=-abb89(47)*abb89(40)
      abb89(16)=abb89(16)+abb89(49)+abb89(32)
      abb89(16)=spbk2e1*abb89(16)
      abb89(17)=abb89(7)*abb89(11)*abb89(17)
      abb89(32)=abb89(51)*abb89(44)
      abb89(17)=abb89(17)+abb89(32)
      abb89(32)=abb89(17)*abb89(52)
      abb89(32)=-2.0_ki*abb89(43)+abb89(32)
      abb89(40)=spbe2e1*abb89(54)
      abb89(40)=abb89(40)+abb89(50)
      abb89(43)=-abb89(45)*abb89(53)
      abb89(39)=spak1k2*abb89(39)
      abb89(37)=abb89(37)+abb89(39)
      abb89(30)=abb89(30)*abb89(37)
      abb89(28)=spak1l4*abb89(28)
      abb89(28)=abb89(28)+abb89(30)
      abb89(18)=spbk2k1*abb89(18)
      abb89(15)=abb89(18)-abb89(15)
      abb89(15)=abb89(27)*abb89(15)
      abb89(18)=-abb89(35)*abb89(23)
      abb89(23)=-abb89(36)*abb89(38)
      abb89(14)=abb89(29)*abb89(14)
      abb89(21)=abb89(29)*abb89(21)
      R2d89=0.0_ki
      rat2 = rat2 + R2d89
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='89' value='", &
          & R2d89, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd89h8
