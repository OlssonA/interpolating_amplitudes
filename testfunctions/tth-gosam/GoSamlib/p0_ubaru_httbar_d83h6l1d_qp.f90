module     p0_ubaru_httbar_d83h6l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity6d83h6l1d_qp.f90
   ! generator: buildfortran_d.py
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_util_qp, only: cond, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, private :: iv0
   integer, private :: iv1
   integer, private :: iv2
   integer, private :: iv3
   real(ki), dimension(4), private :: qshift
   public :: derivative
contains
!---#[ function brack_1:
   pure function brack_1(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd83h6_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(35) :: acd83
      complex(ki) :: brack
      acd83(1)=dotproduct(k2,qshift)
      acd83(2)=abb83(26)
      acd83(3)=dotproduct(qshift,qshift)
      acd83(4)=abb83(13)
      acd83(5)=dotproduct(qshift,spvak2k1)
      acd83(6)=abb83(6)
      acd83(7)=abb83(17)
      acd83(8)=dotproduct(l5,qshift)
      acd83(9)=abb83(14)
      acd83(10)=abb83(22)
      acd83(11)=abb83(8)
      acd83(12)=dotproduct(qshift,spvak1k2)
      acd83(13)=abb83(12)
      acd83(14)=abb83(11)
      acd83(15)=abb83(7)
      acd83(16)=dotproduct(qshift,spvak2l4)
      acd83(17)=dotproduct(qshift,spval4k2)
      acd83(18)=abb83(27)
      acd83(19)=abb83(31)
      acd83(20)=dotproduct(qshift,spvak2l5)
      acd83(21)=dotproduct(qshift,spval5k1)
      acd83(22)=abb83(16)
      acd83(23)=abb83(30)
      acd83(24)=dotproduct(qshift,spval5l4)
      acd83(25)=abb83(29)
      acd83(26)=-acd83(3)*acd83(4)
      acd83(27)=acd83(5)*acd83(6)
      acd83(28)=acd83(1)*acd83(2)
      acd83(26)=acd83(28)+acd83(27)-acd83(7)+acd83(26)
      acd83(26)=acd83(1)*acd83(26)
      acd83(27)=acd83(12)*acd83(13)
      acd83(28)=-acd83(3)*acd83(10)
      acd83(27)=acd83(28)-acd83(14)+acd83(27)
      acd83(27)=acd83(5)*acd83(27)
      acd83(28)=-acd83(10)*acd83(21)
      acd83(28)=acd83(28)-acd83(22)
      acd83(28)=acd83(20)*acd83(28)
      acd83(29)=-acd83(13)*acd83(17)
      acd83(29)=acd83(29)-acd83(18)
      acd83(29)=acd83(16)*acd83(29)
      acd83(30)=-acd83(24)*acd83(25)
      acd83(31)=-acd83(8)*acd83(9)
      acd83(32)=-acd83(21)*acd83(23)
      acd83(33)=-acd83(17)*acd83(19)
      acd83(34)=-acd83(12)*acd83(15)
      acd83(35)=acd83(3)*acd83(11)
      brack=acd83(26)+acd83(27)+acd83(28)+acd83(29)+acd83(30)+acd83(31)+acd83(3&
      &2)+acd83(33)+acd83(34)+acd83(35)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd83h6_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(45) :: acd83
      complex(ki) :: brack
      acd83(1)=k2(iv1)
      acd83(2)=dotproduct(k2,qshift)
      acd83(3)=abb83(26)
      acd83(4)=dotproduct(qshift,qshift)
      acd83(5)=abb83(13)
      acd83(6)=dotproduct(qshift,spvak2k1)
      acd83(7)=abb83(6)
      acd83(8)=abb83(17)
      acd83(9)=l5(iv1)
      acd83(10)=abb83(14)
      acd83(11)=qshift(iv1)
      acd83(12)=abb83(22)
      acd83(13)=abb83(8)
      acd83(14)=spvak2k1(iv1)
      acd83(15)=dotproduct(qshift,spvak1k2)
      acd83(16)=abb83(12)
      acd83(17)=abb83(11)
      acd83(18)=spvak1k2(iv1)
      acd83(19)=abb83(7)
      acd83(20)=spvak2l4(iv1)
      acd83(21)=dotproduct(qshift,spval4k2)
      acd83(22)=abb83(27)
      acd83(23)=spval4k2(iv1)
      acd83(24)=dotproduct(qshift,spvak2l4)
      acd83(25)=abb83(31)
      acd83(26)=spvak2l5(iv1)
      acd83(27)=dotproduct(qshift,spval5k1)
      acd83(28)=abb83(16)
      acd83(29)=spval5k1(iv1)
      acd83(30)=dotproduct(qshift,spvak2l5)
      acd83(31)=abb83(30)
      acd83(32)=spval5l4(iv1)
      acd83(33)=abb83(29)
      acd83(34)=acd83(29)*acd83(30)
      acd83(35)=acd83(26)*acd83(27)
      acd83(36)=2.0_ki*acd83(11)
      acd83(37)=acd83(6)*acd83(36)
      acd83(38)=acd83(14)*acd83(4)
      acd83(34)=acd83(38)+acd83(37)+acd83(34)+acd83(35)
      acd83(34)=acd83(12)*acd83(34)
      acd83(35)=acd83(23)*acd83(24)
      acd83(37)=acd83(20)*acd83(21)
      acd83(38)=-acd83(6)*acd83(18)
      acd83(35)=acd83(38)+acd83(35)+acd83(37)
      acd83(35)=acd83(16)*acd83(35)
      acd83(37)=acd83(4)*acd83(5)
      acd83(38)=-acd83(6)*acd83(7)
      acd83(39)=2.0_ki*acd83(2)
      acd83(39)=-acd83(3)*acd83(39)
      acd83(37)=acd83(39)+acd83(38)+acd83(8)+acd83(37)
      acd83(37)=acd83(1)*acd83(37)
      acd83(38)=acd83(2)*acd83(5)
      acd83(38)=acd83(38)-acd83(13)
      acd83(36)=acd83(36)*acd83(38)
      acd83(38)=-acd83(2)*acd83(7)
      acd83(39)=-acd83(16)*acd83(15)
      acd83(38)=acd83(39)+acd83(17)+acd83(38)
      acd83(38)=acd83(14)*acd83(38)
      acd83(39)=acd83(32)*acd83(33)
      acd83(40)=acd83(9)*acd83(10)
      acd83(41)=acd83(29)*acd83(31)
      acd83(42)=acd83(26)*acd83(28)
      acd83(43)=acd83(23)*acd83(25)
      acd83(44)=acd83(20)*acd83(22)
      acd83(45)=acd83(18)*acd83(19)
      brack=acd83(34)+acd83(35)+acd83(36)+acd83(37)+acd83(38)+acd83(39)+acd83(4&
      &0)+acd83(41)+acd83(42)+acd83(43)+acd83(44)+acd83(45)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd83h6_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(31) :: acd83
      complex(ki) :: brack
      acd83(1)=d(iv1,iv2)
      acd83(2)=dotproduct(k2,qshift)
      acd83(3)=abb83(13)
      acd83(4)=dotproduct(qshift,spvak2k1)
      acd83(5)=abb83(22)
      acd83(6)=abb83(8)
      acd83(7)=k2(iv1)
      acd83(8)=k2(iv2)
      acd83(9)=abb83(26)
      acd83(10)=qshift(iv2)
      acd83(11)=spvak2k1(iv2)
      acd83(12)=abb83(6)
      acd83(13)=qshift(iv1)
      acd83(14)=spvak2k1(iv1)
      acd83(15)=spvak1k2(iv2)
      acd83(16)=abb83(12)
      acd83(17)=spvak1k2(iv1)
      acd83(18)=spvak2l4(iv1)
      acd83(19)=spval4k2(iv2)
      acd83(20)=spvak2l4(iv2)
      acd83(21)=spval4k2(iv1)
      acd83(22)=spvak2l5(iv1)
      acd83(23)=spval5k1(iv2)
      acd83(24)=spvak2l5(iv2)
      acd83(25)=spval5k1(iv1)
      acd83(26)=acd83(15)*acd83(14)
      acd83(27)=acd83(17)*acd83(11)
      acd83(28)=-acd83(19)*acd83(18)
      acd83(29)=-acd83(21)*acd83(20)
      acd83(26)=acd83(29)+acd83(28)+acd83(27)+acd83(26)
      acd83(26)=acd83(16)*acd83(26)
      acd83(27)=-acd83(7)*acd83(3)
      acd83(28)=-acd83(14)*acd83(5)
      acd83(27)=acd83(27)+acd83(28)
      acd83(27)=acd83(10)*acd83(27)
      acd83(28)=-acd83(8)*acd83(3)
      acd83(29)=-acd83(11)*acd83(5)
      acd83(28)=acd83(28)+acd83(29)
      acd83(28)=acd83(13)*acd83(28)
      acd83(29)=acd83(9)*acd83(8)*acd83(7)
      acd83(27)=acd83(29)+acd83(27)+acd83(28)
      acd83(28)=-acd83(23)*acd83(22)
      acd83(29)=-acd83(25)*acd83(24)
      acd83(28)=acd83(29)+acd83(28)
      acd83(28)=acd83(5)*acd83(28)
      acd83(29)=-acd83(2)*acd83(3)
      acd83(30)=-acd83(4)*acd83(5)
      acd83(29)=acd83(6)+acd83(30)+acd83(29)
      acd83(30)=2.0_ki*acd83(1)
      acd83(29)=acd83(30)*acd83(29)
      acd83(30)=acd83(11)*acd83(7)
      acd83(31)=acd83(14)*acd83(8)
      acd83(30)=acd83(30)+acd83(31)
      acd83(30)=acd83(12)*acd83(30)
      brack=acd83(26)+2.0_ki*acd83(27)+acd83(28)+acd83(29)+acd83(30)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd83h6_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(15) :: acd83
      complex(ki) :: brack
      acd83(1)=d(iv1,iv2)
      acd83(2)=k2(iv3)
      acd83(3)=abb83(13)
      acd83(4)=spvak2k1(iv3)
      acd83(5)=abb83(22)
      acd83(6)=d(iv1,iv3)
      acd83(7)=k2(iv2)
      acd83(8)=spvak2k1(iv2)
      acd83(9)=d(iv2,iv3)
      acd83(10)=k2(iv1)
      acd83(11)=spvak2k1(iv1)
      acd83(12)=acd83(2)*acd83(1)
      acd83(13)=acd83(7)*acd83(6)
      acd83(14)=acd83(10)*acd83(9)
      acd83(12)=acd83(14)+acd83(13)+acd83(12)
      acd83(12)=acd83(3)*acd83(12)
      acd83(13)=acd83(4)*acd83(1)
      acd83(14)=acd83(8)*acd83(6)
      acd83(15)=acd83(11)*acd83(9)
      acd83(13)=acd83(15)+acd83(14)+acd83(13)
      acd83(13)=acd83(5)*acd83(13)
      acd83(12)=acd83(13)+acd83(12)
      brack=2.0_ki*acd83(12)
   end function brack_4
!---#] function brack_4:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3) result(numerator)
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd83h6_qp
      implicit none
      complex(ki), intent(in) :: mu2
      integer, intent(in), optional :: i1
      integer, intent(in), optional :: i2
      integer, intent(in), optional :: i3
      complex(ki) :: numerator
      complex(ki) :: loc
      integer :: t1
      integer :: deg
      complex(ki), dimension(4), parameter :: Q = (/ (0.0_ki,0.0_ki),(0.0_ki,0.&
      &0_ki),(0.0_ki,0.0_ki),(0.0_ki,0.0_ki)/)
      qshift = k2
      numerator = 0.0_ki
      deg = 0
      if(present(i1)) then
          iv1=i1
          deg=1
      else
          iv1=1
      end if
      if(present(i2)) then
          iv2=i2
          deg=2
      else
          iv2=1
      end if
      if(present(i3)) then
          iv3=i3
          deg=3
      else
          iv3=1
      end if
      t1 = 0
      if(deg.eq.0) then
         numerator = cond(epspow.eq.t1,brack_1,Q,mu2)
         return
      end if
      if(deg.eq.1) then
         numerator = cond(epspow.eq.t1,brack_2,Q,mu2)
         return
      end if
      if(deg.eq.2) then
         numerator = cond(epspow.eq.t1,brack_3,Q,mu2)
         return
      end if
      if(deg.eq.3) then
         numerator = cond(epspow.eq.t1,brack_4,Q,mu2)
         return
      end if
   end function derivative
!---#] function derivative:
end module     p0_ubaru_httbar_d83h6l1d_qp
