module     p0_ubaru_httbar_d83h1l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity1d83h1l1d.f90
   ! generator: buildfortran_d.py
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_util, only: cond, d => metric_tensor
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
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd83h1
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(16) :: acd83
      complex(ki) :: brack
      acd83(1)=dotproduct(k2,qshift)
      acd83(2)=dotproduct(qshift,spval5k2)
      acd83(3)=abb83(7)
      acd83(4)=dotproduct(qshift,qshift)
      acd83(5)=abb83(8)
      acd83(6)=dotproduct(qshift,spvak1k2)
      acd83(7)=abb83(5)
      acd83(8)=abb83(6)
      acd83(9)=dotproduct(qshift,spval5k1)
      acd83(10)=abb83(9)
      acd83(11)=dotproduct(qshift,spval4k2)
      acd83(12)=dotproduct(qshift,spval5l4)
      acd83(13)=abb83(14)
      acd83(14)=acd83(3)*acd83(1)
      acd83(15)=-acd83(4)*acd83(5)
      acd83(16)=acd83(7)*acd83(6)
      acd83(14)=-acd83(8)+acd83(16)+acd83(15)+acd83(14)
      acd83(14)=acd83(2)*acd83(14)
      acd83(15)=-acd83(6)*acd83(5)
      acd83(15)=-acd83(10)+acd83(15)
      acd83(15)=acd83(9)*acd83(15)
      acd83(16)=acd83(11)*acd83(5)
      acd83(16)=-acd83(13)+acd83(16)
      acd83(16)=acd83(12)*acd83(16)
      brack=acd83(14)+acd83(15)+acd83(16)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd83h1
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(26) :: acd83
      complex(ki) :: brack
      acd83(1)=k2(iv1)
      acd83(2)=dotproduct(qshift,spval5k2)
      acd83(3)=abb83(7)
      acd83(4)=qshift(iv1)
      acd83(5)=abb83(8)
      acd83(6)=spval5k2(iv1)
      acd83(7)=dotproduct(k2,qshift)
      acd83(8)=dotproduct(qshift,qshift)
      acd83(9)=dotproduct(qshift,spvak1k2)
      acd83(10)=abb83(5)
      acd83(11)=abb83(6)
      acd83(12)=spvak1k2(iv1)
      acd83(13)=dotproduct(qshift,spval5k1)
      acd83(14)=spval5k1(iv1)
      acd83(15)=abb83(9)
      acd83(16)=spval4k2(iv1)
      acd83(17)=dotproduct(qshift,spval5l4)
      acd83(18)=spval5l4(iv1)
      acd83(19)=dotproduct(qshift,spval4k2)
      acd83(20)=abb83(14)
      acd83(21)=acd83(14)*acd83(9)
      acd83(22)=acd83(4)*acd83(2)
      acd83(23)=acd83(8)*acd83(6)
      acd83(24)=acd83(13)*acd83(12)
      acd83(25)=-acd83(17)*acd83(16)
      acd83(26)=-acd83(19)*acd83(18)
      acd83(21)=acd83(26)+acd83(25)+acd83(24)+acd83(23)+2.0_ki*acd83(22)+acd83(&
      &21)
      acd83(21)=acd83(5)*acd83(21)
      acd83(22)=-acd83(12)*acd83(10)
      acd83(23)=-acd83(1)*acd83(3)
      acd83(22)=acd83(23)+acd83(22)
      acd83(22)=acd83(2)*acd83(22)
      acd83(23)=-acd83(10)*acd83(9)
      acd83(24)=-acd83(7)*acd83(3)
      acd83(23)=acd83(11)+acd83(24)+acd83(23)
      acd83(23)=acd83(6)*acd83(23)
      acd83(24)=acd83(15)*acd83(14)
      acd83(25)=acd83(20)*acd83(18)
      brack=acd83(21)+acd83(22)+acd83(23)+acd83(24)+acd83(25)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd83h1
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(24) :: acd83
      complex(ki) :: brack
      acd83(1)=d(iv1,iv2)
      acd83(2)=dotproduct(qshift,spval5k2)
      acd83(3)=abb83(8)
      acd83(4)=k2(iv1)
      acd83(5)=spval5k2(iv2)
      acd83(6)=abb83(7)
      acd83(7)=k2(iv2)
      acd83(8)=spval5k2(iv1)
      acd83(9)=qshift(iv1)
      acd83(10)=qshift(iv2)
      acd83(11)=spvak1k2(iv2)
      acd83(12)=abb83(5)
      acd83(13)=spvak1k2(iv1)
      acd83(14)=spval5k1(iv2)
      acd83(15)=spval5k1(iv1)
      acd83(16)=spval4k2(iv1)
      acd83(17)=spval5l4(iv2)
      acd83(18)=spval4k2(iv2)
      acd83(19)=spval5l4(iv1)
      acd83(20)=acd83(1)*acd83(2)
      acd83(21)=acd83(8)*acd83(10)
      acd83(22)=acd83(5)*acd83(9)
      acd83(20)=acd83(22)+acd83(20)+acd83(21)
      acd83(21)=acd83(18)*acd83(19)
      acd83(22)=acd83(16)*acd83(17)
      acd83(23)=-acd83(13)*acd83(14)
      acd83(24)=-acd83(11)*acd83(15)
      acd83(20)=acd83(24)+acd83(23)+acd83(21)+acd83(22)-2.0_ki*acd83(20)
      acd83(20)=acd83(3)*acd83(20)
      acd83(21)=acd83(11)*acd83(12)
      acd83(22)=acd83(6)*acd83(7)
      acd83(21)=acd83(21)+acd83(22)
      acd83(21)=acd83(8)*acd83(21)
      acd83(22)=acd83(12)*acd83(13)
      acd83(23)=acd83(6)*acd83(4)
      acd83(22)=acd83(22)+acd83(23)
      acd83(22)=acd83(5)*acd83(22)
      brack=acd83(20)+acd83(21)+acd83(22)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd83h1
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(10) :: acd83
      complex(ki) :: brack
      acd83(1)=d(iv1,iv2)
      acd83(2)=spval5k2(iv3)
      acd83(3)=abb83(8)
      acd83(4)=d(iv1,iv3)
      acd83(5)=spval5k2(iv2)
      acd83(6)=d(iv2,iv3)
      acd83(7)=spval5k2(iv1)
      acd83(8)=acd83(2)*acd83(1)
      acd83(9)=acd83(5)*acd83(4)
      acd83(10)=acd83(7)*acd83(6)
      acd83(8)=acd83(10)+acd83(8)+acd83(9)
      brack=2.0_ki*acd83(8)*acd83(3)
   end function brack_4
!---#] function brack_4:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3) result(numerator)
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd83h1
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
end module     p0_ubaru_httbar_d83h1l1d
